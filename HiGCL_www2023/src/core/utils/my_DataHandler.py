import torch
import torch.utils.data as data
import numpy as np
import scipy.sparse
import multiprocessing
import time
import os
import random

def load_data(config):
    multi_beh_data = []
    multi_beh_id = []
    behavior_name = None
    user_item_data_valid = []
    user_item_data_test = []
    
    for index, file_name in enumerate(config['train_file_name']):
        train_file = config['data_dir'] + file_name
        with open(train_file, 'r') as f:
            num_inter = 0
            lines = f.readlines()
            for line in lines:
                line = [int(id) for id in line.strip('\n').split(',')]
                u_id = line[0]
                for i_id in line[1:]:
                    multi_beh_data.append([u_id, i_id])

                num_inter += (len(line) - 1)

            multi_beh_id.extend([index]*num_inter)
    
    multi_beh_data = np.array(multi_beh_data)
    multi_beh_id = np.squeeze(np.array(multi_beh_id).reshape((-1,1)))

    behavior_name = config['behavior_name']

    with open(config['data_dir'] + config['valid_file_name'],'r') as f:
        lines = f.readlines()
        for line in lines:
            line = [int(id) for id in line.strip('\n').split(',')]
            u_id = line[0]
            for i_id in line[1:]:
                user_item_data_valid.append([u_id, i_id])
    user_item_data_valid = np.array(user_item_data_valid)

    with open(config['data_dir'] + config['test_file_name'],'r') as f:
        lines = f.readlines()
        for line in lines:
            line = [int(id) for id in line.strip('\n').split(',')]
            u_id = line[0]
            for i_id in line[1:]:
                user_item_data_test.append([u_id, i_id])
    user_item_data_test = np.array(user_item_data_test)
    
    if os.path.exists(config['data_dir'] + 'sample_valid.txt') and os.path.exists(config['data_dir'] + 'sample_test.txt'):
        sample_valid_mat = np.loadtxt(config['data_dir'] + 'sample_valid.txt', dtype=int, delimiter=',')
        sample_test_mat = np.loadtxt(config['data_dir'] + 'sample_test.txt', dtype=int, delimiter=',')
    else:
        sample_valid_mat = None
        sample_test_mat = None
        
    return multi_beh_data, multi_beh_id, behavior_name, user_item_data_valid, user_item_data_test, sample_valid_mat, sample_test_mat



class RecDataset_beh(data.Dataset):
    def __init__(self, u2u_data, multi_beh_data, multi_beh_id, behavior_id, graph_norm_flag, sparse_graph_flag, directed_flag, graph_include_self,\
        u2u_flag, i2i_flag, p_cnt, target_behavior_id, num_users, num_items, folds):  
        """
        params:
            u2u_data: [num_relations, 2]  each row is a follow / friend relationship  (user1_id, user2_id)     +++ numpy array +++
            multi_beh_data: [num_interactions, 2]   each row is a user - item interaction data  (user_id, item_id)
            multi_beh_id: [num_interactions]   each element is a behavior id, indicating click, like, dislike etc.
        """
        super(RecDataset_beh, self).__init__()
        self.u2u_data = u2u_data
        self.multi_beh_data = multi_beh_data
        self.multi_beh_id = multi_beh_id
        self.behavior_id = behavior_id
        self.graph_norm_flag = graph_norm_flag
        self.sparse_graph_flag = sparse_graph_flag
        self.directed_flag = directed_flag
        self.graph_include_self = graph_include_self
        self.num_users = num_users
        self.num_items = num_items
        self.multi_beh_data_sp_matrix = [] 

        self.sample_pair = None    # sampled pairs for BPR Loss
        self.sample_len = -1

        self.u2u_flag = u2u_flag
        self.i2i_flag = i2i_flag

        self.p_cnt = p_cnt
        self.target_behavior_id = target_behavior_id
        self.folds = folds

        self.allpos = []

        self.target_behavior_dense_matrix = None
        # preprocess u2u_data, multi_beh_data, multi_beh_id ---->> u2u_adj, multi_u2i_adj, i2i_adj
        self.my_dic = multiprocessing.Manager().dict()
        self.multi_adjs = self.generate_adjs()
        self.pre_for_sample()


    def get_allpos(self, sp_graph):
        posItems = []
        for user in range(self.num_users):
            posItems.append(sp_graph[user].nonzero()[1])
        
        self.allpos.append(posItems)


    def generate_adjs(self):
        """
        generate adjacency matrixs from u2u_data, multi_beh_data, multi_beh_id data
        """
        # get all user and item id
        if self.u2u_flag:
            all_users = np.unique(np.hstack(self.u2u_data.reshape((1,-1)), self.multi_beh_data[:,0].reshape((1,-1))))
        else:
            all_users = np.unique(self.multi_beh_data[:,0])

        all_items = np.unique(self.multi_beh_data[:,1])
        print("user, item=",self.num_users, self.num_items)

        # ----------------------------------- generte scipy.sparse coo_matrix adj -----------------------------------
        # generate u2u_adj
        u2u_adj = None
        if self.u2u_flag:
            u2u_row = self.u2u_data[:,0]
            u2u_col = self.u2u_data[:,1]
            u2u_val = np.ones(len(u2u_row))
            u2u_adj = scipy.sparse.coo_matrix((u2u_val, (u2u_row, u2u_col)), shape=(len(all_users), len(all_users)))
            if self.graph_include_self:
                row, col = np.diag_indices_from(u2u_adj)
                u2u_adj[row,col] = 1



        # generate multi_u2i_adj 
        existing_user_item_list = []   
        multi_u2i_adj = []
        for i in range(len(self.behavior_id)):
            cur_beh_user_item = []
            cur_beh_ids = np.where(self.multi_beh_id == i)[0]
            cur_u_ids = self.multi_beh_data[cur_beh_ids, 0]
            cur_i_ids = self.multi_beh_data[cur_beh_ids, 1]
            
            existing_user = np.unique(cur_u_ids)
            existing_item = np.unique(cur_i_ids)
            cur_beh_user_item.append(torch.from_numpy(existing_user))
            cur_beh_user_item.append(torch.from_numpy(existing_item))
            existing_user_item_list.append(cur_beh_user_item)
            

            cur_u2i_val = np.ones(len(cur_u_ids))
            cur_u2i_adj = scipy.sparse.coo_matrix((cur_u2i_val, (cur_u_ids, cur_i_ids)), shape=(self.num_users, self.num_items))
            multi_u2i_adj.append(cur_u2i_adj)
            self.multi_beh_data_sp_matrix.append(self.trans_to_torch_sparse(cur_u2i_adj.copy()))
            self.get_allpos(cur_u2i_adj.tocsr())

            if i==self.target_behavior_id:
                self.target_behavior_dense_matrix = self.multi_beh_data_sp_matrix[i].to_dense()
               
                
        # generate i2i_adj
        i2i_adj = None
        if self.i2i_flag:
            all_u2i_val = np.ones(len(self.multi_beh_data[:,0]))
            all_u2i_matrix = scipy.sparse.coo_matrix((all_u2i_val, (self.multi_beh_data[:,0], self.multi_beh_data[:,1])), shape=(len(all_users), len(all_items)))
            i2i_adj = all_u2i_matrix.transpose().dot(all_u2i_matrix)
            i2i_adj[i2i_adj>0] = 1
            row, col = np.diag_indices_from(i2i_adj)
            if self.graph_include_self:
                i2i_adj[row, col] = 1
            else:
                i2i_adj[row, col] = 0

        
        # ----------------------------------- post process generated adjacency matrixs -----------------------------------
        if not self.directed_flag:      # undirected graph
            if self.u2u_flag:
                u2u_adj = u2u_adj + u2u_adj.transpose()
                u2u_adj[u2u_adj > 0] = 1

            if self.i2i_flag:
                i2i_adj = i2i_adj + i2i_adj.transpose()
                i2i_adj[i2i_adj>0] = 1
        

        if self.graph_norm_flag:    # graph normalization
            if self.u2u_flag:
                u2u_adj = self.normalize_sparse_adj(u2u_adj)
            
            for index, u2i_adj in enumerate(multi_u2i_adj):
                multi_u2i_adj[index] = self.normalize_sparse_adj(u2i_adj).tocoo()
            
            if self.i2i_flag:
                i2i_adj = self.normalize_sparse_adj(i2i_adj)
        

        multi_u2i_fold_list = []
        multi_i2u_fold_list = []
        for u2i_adj in multi_u2i_adj:
            beh_u2i_fold_list = []
            u2i_adj_csr = u2i_adj.tocsr()
            fold_len_u2i = self.num_users // self.folds
            for i_fold in range(self.folds):
                start = i_fold * fold_len_u2i
                if i_fold == self.folds-1:
                    end = self.num_users
                else:
                    end = (i_fold+1)*fold_len_u2i
                beh_u2i_fold_list.append(self._convert_sp_mat_to_sp_tensor(u2i_adj_csr[start:end]))
            multi_u2i_fold_list.append(beh_u2i_fold_list)

            beh_i2u_fold_list = []
            i2u_adj_csr = u2i_adj.transpose().tocsr()
            fold_len_i2u = self.num_items // self.folds
            for i_fold in range(self.folds):
                start = i_fold * fold_len_i2u
                if i_fold == self.folds-1:
                    end = self.num_items
                else:
                    end = (i_fold+1)*fold_len_i2u
                beh_i2u_fold_list.append(self._convert_sp_mat_to_sp_tensor(i2u_adj_csr[start:end]))
            multi_i2u_fold_list.append(beh_i2u_fold_list)


        else:
            if self.u2u_flag:
                u2u_adj= torch.from_numpy(u2u_adj.toarray())

            for u2i_adj in multi_u2i_adj:
                u2i_adj = torch.from_numpy(u2i_adj.toarray())
            
            if self.i2i_flag:
                i2i_adj = torch.from_numpy(i2i_adj.toarray())


        multi_adjs = {}
        multi_adjs['multi_u2i_fold_list'] = multi_u2i_fold_list
        multi_adjs['multi_i2u_fold_list'] = multi_i2u_fold_list
        multi_adjs['existing_user_item_list'] = existing_user_item_list


        return multi_adjs


    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


    def trans_to_torch_sparse(self, scipy_sparse_matrix):
        # convert scipy.sparse matrix to torch.sparse
        indices = torch.from_numpy(np.vstack((scipy_sparse_matrix.row, scipy_sparse_matrix.col)).astype(np.int64))  
        values = torch.from_numpy(scipy_sparse_matrix.data)  
        shape = torch.Size(scipy_sparse_matrix.shape)
        torch_sparse_matrix = torch.sparse_coo_tensor(indices, values, shape).to(torch.float32)

        return torch_sparse_matrix



    def normalize_sparse_adj(self, adj):
        """normalize sparse matrix: symmetric normalized Laplacian"""
        rowsum = np.array(adj.sum(1))
        r_inv_sqrt = np.power(rowsum+1e-20, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = scipy.sparse.diags(r_inv_sqrt)

        colsum = np.array(adj.sum(0))
        c_inv_sqrt = np.power(colsum+1e-20, -0.5).flatten()
        c_inv_sqrt[np.isinf(c_inv_sqrt)] = 0.
        c_mat_inv_sqrt = scipy.sparse.diags(c_inv_sqrt)
        
        return r_mat_inv_sqrt.dot(adj).dot(c_mat_inv_sqrt)

    
    def pre_for_sample(self):
        self.sample_len = -1     # sample_len = the maximum number of different behavior
        for i in range(len(self.behavior_id)):
            if self.behavior_id[i] == self.target_behavior_id:
                cur_beh_ids = np.where(self.multi_beh_id == self.behavior_id[i])[0]
                self.sample_len = len(cur_beh_ids)
            
        self.sample_pair = torch.zeros((len(self.behavior_id), self.sample_len, 3))       # [num_beh, sample_len, 3]   


    
    def sub_sample(self, begin, end, user, multi_beh_data_sp_matrix, num_items):
        sample_list = []
        for j in range(begin, end):
            sample_neg = random.randint(0, num_items - 1)
            while multi_beh_data_sp_matrix[int(user[j]), int(sample_neg)] == 1:
                sample_neg = random.randint(0, num_items-1)
            
            sample_list.append(sample_neg)
        
        self.my_dic[begin] = sample_list



    def ng_sample(self):
        """
        sample (u, observed_item, unobserved_item) pairs for BPR Loss
        """
        beg = time.time()
        sample_pair = np.zeros((len(self.behavior_id), self.sample_len, 3))       # [num_beh, sample_len, 3]
        target_ids = np.where(self.multi_beh_id == self.target_behavior_id)[0]
        target_users = self.multi_beh_data[target_ids][:,0]

        for i in range(len(self.behavior_id)):
            #-------------------------------------------------------------------
            sample_pair[i,:,0] = target_users.copy()
            for j in range(self.sample_len):
                cur_user = target_users[j]
                posForUser = self.allpos[i][cur_user]
                if len(posForUser) == 0:
                    sample_pair[i,j,1] = -1
                else:
                    posindex = np.random.randint(0, len(posForUser))
                    positem = posForUser[posindex]
                    sample_pair[i,j,1] = positem
            #-------------------------------------------------------------------

            p_cnt = self.p_cnt
            interval = int(self.sample_len/p_cnt)
            p_list = []
            multi_beh_data_sp_matrix = self.multi_beh_data_sp_matrix[i].to_dense()
            for k in range(p_cnt):
                if k < p_cnt-1:
                    p_list.append(multiprocessing.Process(target=self.sub_sample, args=(k*interval, (k+1)*interval,sample_pair[i,:,0], multi_beh_data_sp_matrix, self.num_items)))
                else:
                    p_list.append(multiprocessing.Process(target=self.sub_sample, args=(k*interval, self.sample_len,sample_pair[i,:,0], multi_beh_data_sp_matrix, self.num_items)))
            
            for k in range(p_cnt):
                p_list[k].start()
            
            for k in range(p_cnt):
                p_list[k].join()

            for key in self.my_dic:
                val = self.my_dic[key]
                val_len = len(val)
                sample_pair[i,key:key+val_len,2] = np.array(val)

        end = time.time()
        print("sample time cost:", end-beg)
        
        self.sample_pair = torch.from_numpy(sample_pair)


        

    def __len__(self):
        return self.sample_pair.shape[1]


    def __getitem__(self, idx):
        user = self.sample_pair[:, idx, 0]
        observed_item = self.sample_pair[:, idx, 1]
        unobserved_item = self.sample_pair[:, idx, 2]

        return user, observed_item, unobserved_item






class RecDataset_test(data.Dataset):
    def __init__(self, num_users, num_items, user_item_data_test, sample_test_mat):  
        """
        params:
        user_item_data_test: [num_test_interaction, 2]
        """
        super(RecDataset_test, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.user_item_data_test = user_item_data_test
        self.user_item_data_test_sparse = None
        self.test_users = None
        self.sample_test_mat = sample_test_mat
        self.init_adjs = self.generate_test_u2i_adj()
        
    def generate_test_u2i_adj(self):
        # generate multi_u2i_adj    

        u_ids = self.user_item_data_test[:, 0]
        i_ids = self.user_item_data_test[:, 1]
        u2i_val = np.ones(len(u_ids))
        u2i_adj = scipy.sparse.coo_matrix((u2i_val, (u_ids, i_ids)), shape=(self.num_users, self.num_items))
        self.user_item_data_test_sparse = self.trans_to_torch_sparse(u2i_adj.copy())

        self.test_users = np.unique(u_ids)


    def trans_to_torch_sparse(self, scipy_sparse_matrix):
        # convert scipy.sparse matrix to torch.sparse
        indices = torch.from_numpy(np.vstack((scipy_sparse_matrix.row, scipy_sparse_matrix.col)).astype(np.int64))  
        values = torch.from_numpy(scipy_sparse_matrix.data)  
        shape = torch.Size(scipy_sparse_matrix.shape)
        torch_sparse_matrix = torch.sparse_coo_tensor(indices, values, shape).to(torch.float32)

        return torch_sparse_matrix


    def __len__(self):
        return len(self.test_users)


    def __getitem__(self, idx):
        return self.test_users[idx], self.user_item_data_test_sparse[int(self.test_users[idx])].to_dense()

