import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch_sparse
import time


class new_Multi_Encoder(nn.Module):
    def __init__(self, config, mlp_flag, device, mask_flag, num_users, num_items, dim_each_feedback, n_layers, behavior_names, behavior_type, sparse_graph_flag, combine_choice = 'mean', acti_func = 'ReLU', graph_module = "LightGCN", \
        u2u_flag = True, i2i_flag = True, batch_norm = False, dropout = 0, denoise_flag = True):
        """
        Multi-behavior / type encoder
        params:
            num_users: number of users
            num_items: number of items
            embed_dim: dimension of embedding
            n_layers: number of GCN layers
            beh_names: description of each behavior, such as like, dislike, view etc.
            behavior_type: K x 1 , indicating the type (EP/0, IP/1, EN/2, IN/3) of each behavior
            graph_module: which type of graph encoder to use (LightGCN, GCN, GAT ...)
        """
        super(new_Multi_Encoder, self).__init__()
        self.config = config
        self.mlp_flag = mlp_flag
        self.device = device
        self.num_users = num_users
        self.num_items = num_items
        self.dim_each_feedback = dim_each_feedback
        self.n_layers = n_layers
        self.behavior_names = behavior_names
        self.behavior_type = torch.tensor(behavior_type)
        self.graph_module = graph_module
        self.act_func = nn.PReLU() if acti_func == 'PReLU' else nn.ReLU()
        self.combine_choice = combine_choice
        self.sparse_graph_flag = sparse_graph_flag
        self.u2u_flag = u2u_flag
        self.i2i_flag = i2i_flag
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.denoise_flag = denoise_flag
        self.mask_flag = mask_flag

        # prepare the learnable parameters of the model
        self.weight_dic = self.init_weight()

        self.encoder = LightGCN(self.num_users, self.num_items, self.dim_each_feedback, self.n_layers, self.sparse_graph_flag,\
                    self.u2u_flag, self.i2i_flag, self.batch_norm, self.dropout)
        

    def init_weight(self):
        EP_size = len(torch.squeeze((self.behavior_type == 0).nonzero(), dim = -1))
        IP_size = len(torch.squeeze((self.behavior_type == 1).nonzero(), dim = -1))
        EN_size = len(torch.squeeze((self.behavior_type == 2).nonzero(), dim = -1))
        IN_size = len(torch.squeeze((self.behavior_type == 3).nonzero(), dim = -1))

        fine_weight_input_EP = EP_size * self.dim_each_feedback if self.combine_choice == 'cat' else self.dim_each_feedback
        fine_weight_input_IP = IP_size * self.dim_each_feedback if self.combine_choice == 'cat' else self.dim_each_feedback
        fine_weight_input_EN = EN_size * self.dim_each_feedback if self.combine_choice == 'cat' else self.dim_each_feedback
        fine_weight_input_IN = IN_size * self.dim_each_feedback if self.combine_choice == 'cat' else self.dim_each_feedback

        # coarse_weight_input = 4 * self.embed_dim if self.combine_choice == 'cat' else self.embed_dim

            
        # user weight
        fine_weight_EP_u = nn.Parameter(torch.Tensor(fine_weight_input_EP, self.dim_each_feedback), requires_grad=True)
        fine_weight_IP_u = nn.Parameter(torch.Tensor(fine_weight_input_IP, self.dim_each_feedback), requires_grad=True)
        fine_weight_EN_u = nn.Parameter(torch.Tensor(fine_weight_input_EN, self.dim_each_feedback), requires_grad=True)
        fine_weight_IN_u = nn.Parameter(torch.Tensor(fine_weight_input_IN, self.dim_each_feedback), requires_grad=True)
        pos_weight_u = nn.Parameter(torch.Tensor(self.dim_each_feedback, self.dim_each_feedback), requires_grad=True)
        neg_weight_u = nn.Parameter(torch.Tensor(self.dim_each_feedback, self.dim_each_feedback), requires_grad=True)
        # coarse_weight_u = nn.Parameter(torch.Tensor(coarse_weight_input, self.embed_dim))
        init.xavier_uniform_(fine_weight_EP_u)
        init.xavier_uniform_(fine_weight_IP_u)
        init.xavier_uniform_(fine_weight_EN_u)
        init.xavier_uniform_(fine_weight_IN_u)
        init.xavier_uniform_(pos_weight_u)
        init.xavier_uniform_(neg_weight_u)
        # init.xavier_uniform_(coarse_weight_u)

        # item weight
        fine_weight_EP_v = nn.Parameter(torch.Tensor(fine_weight_input_EP, self.dim_each_feedback), requires_grad=True)
        fine_weight_IP_v = nn.Parameter(torch.Tensor(fine_weight_input_IP, self.dim_each_feedback), requires_grad=True)
        fine_weight_EN_v = nn.Parameter(torch.Tensor(fine_weight_input_EN, self.dim_each_feedback), requires_grad=True)
        fine_weight_IN_v = nn.Parameter(torch.Tensor(fine_weight_input_IN, self.dim_each_feedback), requires_grad=True)
        pos_weight_v = nn.Parameter(torch.Tensor(self.dim_each_feedback, self.dim_each_feedback), requires_grad=True)
        neg_weight_v = nn.Parameter(torch.Tensor(self.dim_each_feedback, self.dim_each_feedback), requires_grad=True)
        # coarse_weight_v = nn.Parameter(torch.Tensor(coarse_weight_input, self.embed_dim))
        init.xavier_uniform_(fine_weight_EP_v)
        init.xavier_uniform_(fine_weight_IP_v)
        init.xavier_uniform_(fine_weight_EN_v)
        init.xavier_uniform_(fine_weight_IN_v)
        init.xavier_uniform_(pos_weight_v)
        init.xavier_uniform_(neg_weight_v)
        # init.xavier_uniform_(coarse_weight_v)

        weight_dic = {}
        weight_dic["fine_weight_EP_u"] = fine_weight_EP_u
        weight_dic["fine_weight_IP_u"] = fine_weight_IP_u
        weight_dic["fine_weight_EN_u"] = fine_weight_EN_u
        weight_dic["fine_weight_IN_u"] = fine_weight_IN_u
        weight_dic["pos_weight_u"] = pos_weight_u
        weight_dic["neg_weight_u"] = neg_weight_u
        # weight_dic["coarse_weight_u"] = coarse_weight_u

        weight_dic["fine_weight_EP_v"] = fine_weight_EP_v
        weight_dic["fine_weight_IP_v"] = fine_weight_IP_v
        weight_dic["fine_weight_EN_v"] = fine_weight_EN_v
        weight_dic["fine_weight_IN_v"] = fine_weight_IN_v
        weight_dic["pos_weight_v"] = pos_weight_v
        weight_dic["neg_weight_v"] = neg_weight_v
        # weight_dic["coarse_weight_v"] = coarse_weight_v
        
        weight_dic = nn.ParameterDict(weight_dic)
        return weight_dic

        
        

    def forward(self, multi_adjs, init_user_embeddings = None, init_item_embeddings = None):
        """
        params:
            adj_matrixs: dic_type, u2u_adj, i2i_adj, multi_u2i_adj (list)
            init_user_embeddings: pre-trained initial user embeddings
            init_item_embeddings: pre-trained initial item embeddings
        
        return:
            user(item)_embeddings_each_behavior: fine-grained user/item embedding 
            user(item)_embeddings_each_feedback: coarse-grained user/item embedding
            user(item)_embedding: final user/item embedding used by the downstream task
        """
        user_embeddings_each_behavior = []
        item_embeddings_each_behavior = []
        
        # self.prepare_half_sparse(adj_matrixs['multi_u2i_adj'])

        # --------------- forward --------------- 
        if self.graph_module == "LightGCN":
            for i in range(len(self.behavior_type)):
                user_embedding, item_embedding = self.encoder(None, multi_adjs['multi_u2i_fold_list'][i], multi_adjs['multi_i2u_fold_list'][i], multi_adjs['existing_user_item_list'][i], None, self.mask_flag, self.device)
                user_embeddings_each_behavior.append(user_embedding)
                item_embeddings_each_behavior.append(item_embedding)

        user_embeddings_each_behavior = torch.stack(user_embeddings_each_behavior, dim = 0)
        item_embeddings_each_behavior = torch.stack(item_embeddings_each_behavior, dim = 0)


        # --------------- combine behavior embeddings to form type-aware feedback embedding --------------- 
        user_embeddings_each_feedback = []
        item_embeddings_each_feedback = []

        if self.combine_choice == 'mean':
            EP_index = torch.squeeze((self.behavior_type == 0).nonzero(), dim = -1)
            if len(EP_index) > 0:
                if self.mlp_flag:
                    EP_embeddings_user = self.act_func(torch.matmul(torch.mean(user_embeddings_each_behavior[EP_index], dim = 0), self.weight_dic["fine_weight_EP_u"]))
                    EP_embeddings_item = self.act_func(torch.matmul(torch.mean(item_embeddings_each_behavior[EP_index], dim = 0), self.weight_dic["fine_weight_EP_v"]))
                else:
                    EP_embeddings_user = torch.mean(user_embeddings_each_behavior[EP_index], dim = 0)
                    EP_embeddings_item = torch.mean(item_embeddings_each_behavior[EP_index], dim = 0)

            IP_index = torch.squeeze((self.behavior_type == 1).nonzero(), dim = -1)
            if len(IP_index) > 0:
                if self.mlp_flag:
                    IP_embeddings_user = self.act_func(torch.matmul(torch.mean(user_embeddings_each_behavior[IP_index], dim = 0), self.weight_dic["fine_weight_IP_u"]))
                    IP_embeddings_item = self.act_func(torch.matmul(torch.mean(item_embeddings_each_behavior[IP_index], dim = 0), self.weight_dic["fine_weight_IP_v"]))
                else:
                    IP_embeddings_user = torch.mean(user_embeddings_each_behavior[IP_index], dim = 0)
                    IP_embeddings_item = torch.mean(item_embeddings_each_behavior[IP_index], dim = 0)

            EN_index = torch.squeeze((self.behavior_type == 2).nonzero(), dim = -1)
            if len(EN_index) > 0:
                if self.mlp_flag:
                    EN_embeddings_user = self.act_func(torch.matmul(torch.mean(user_embeddings_each_behavior[EN_index], dim = 0), self.weight_dic["fine_weight_EN_u"]))
                    EN_embeddings_item = self.act_func(torch.matmul(torch.mean(item_embeddings_each_behavior[EN_index], dim = 0), self.weight_dic["fine_weight_EN_v"]))
                else:
                    EN_embeddings_user = torch.mean(user_embeddings_each_behavior[EN_index], dim = 0)
                    EN_embeddings_item = torch.mean(item_embeddings_each_behavior[EN_index], dim = 0)


            IN_index = torch.squeeze((self.behavior_type == 3).nonzero(), dim = -1)
            if len(IN_index) > 0:
                if self.mlp_flag:
                    IN_embeddings_user = self.act_func(torch.matmul(torch.mean(user_embeddings_each_behavior[IN_index], dim = 0), self.weight_dic["fine_weight_IN_u"]))
                    IN_embeddings_item = self.act_func(torch.matmul(torch.mean(item_embeddings_each_behavior[IN_index], dim = 0), self.weight_dic["fine_weight_IN_v"]))
                else:
                    IN_embeddings_user = torch.mean(user_embeddings_each_behavior[IN_index], dim = 0)
                    IN_embeddings_item = torch.mean(item_embeddings_each_behavior[IN_index], dim = 0)



            # --------------- denoise IP and IN by orthogonal mapping --------------- 
            if self.denoise_flag:
                # denoise IP with EN
                if len(IP_index) > 0 and len(EN_index) > 0:
                    IP_embeddings_user = self.orthogonal_denoise(IP_embeddings_user, EN_embeddings_user)
                    IP_embeddings_item = self.orthogonal_denoise(IP_embeddings_item, EN_embeddings_item)
                # denoise IN with EP
                if len(IN_index) > 0 and len(EP_index) > 0:
                    IN_embeddings_user = self.orthogonal_denoise(IN_embeddings_user, EP_embeddings_user)
                    IN_embeddings_item = self.orthogonal_denoise(IN_embeddings_item, EP_embeddings_item)

            if len(EP_index) > 0: 
                user_embeddings_each_feedback.append(EP_embeddings_user)
                item_embeddings_each_feedback.append(EP_embeddings_item)
         
            if len(IP_index) > 0:
                user_embeddings_each_feedback.append(IP_embeddings_user)
                item_embeddings_each_feedback.append(IP_embeddings_item)
  
            if len(EN_index) > 0:
                user_embeddings_each_feedback.append(EN_embeddings_user)
                item_embeddings_each_feedback.append(EN_embeddings_item)

            if len(IN_index) > 0:
                user_embeddings_each_feedback.append(IN_embeddings_user)
                item_embeddings_each_feedback.append(IN_embeddings_item)
             
            

            # --------------- combine feedback embeddings to form the final user and item embedding --------------- 
            user_embeddings_each_feedback = torch.stack(user_embeddings_each_feedback, dim = 0)
            item_embeddings_each_feedback = torch.stack(item_embeddings_each_feedback, dim = 0)

            # 把feedback embeddings 融合成 positive 和 negative embedding
            pos_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type']) <= 1).nonzero(), dim = -1)
            neg_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type']) > 1).nonzero(), dim = -1)


            if self.mlp_flag:
                if len(pos_index) > 0:
                    #pos_embeddings_user = self.act_func(torch.matmul(torch.mean(user_embeddings_each_feedback[pos_index], dim = 0), self.weight_dic["pos_weight_u"]))
                    #pos_embeddings_item = self.act_func(torch.matmul(torch.mean(item_embeddings_each_feedback[pos_index], dim = 0), self.weight_dic["pos_weight_v"]))
                    pos_embeddings_user = torch.mean(user_embeddings_each_feedback[pos_index], dim = 0)
                    pos_embeddings_item = torch.mean(item_embeddings_each_feedback[pos_index], dim = 0)

                if len(neg_index) > 0:
                    neg_embeddings_user = self.act_func(torch.matmul(torch.mean(user_embeddings_each_feedback[neg_index], dim = 0), self.weight_dic["neg_weight_u"]))
                    neg_embeddings_item = self.act_func(torch.matmul(torch.mean(item_embeddings_each_feedback[neg_index], dim = 0), self.weight_dic["neg_weight_v"]))
            else:
                if len(pos_index) > 0:
                    pos_embeddings_user = torch.mean(user_embeddings_each_feedback[pos_index], dim = 0)
                    pos_embeddings_item = torch.mean(item_embeddings_each_feedback[pos_index], dim = 0)
                if len(neg_index) > 0:
                    neg_embeddings_user = torch.mean(user_embeddings_each_feedback[neg_index], dim = 0)
                    neg_embeddings_item = torch.mean(item_embeddings_each_feedback[neg_index], dim = 0)

            user_embeddings_pos_neg = []
            item_embeddings_pos_neg = []
            if len(pos_index) > 0:
                user_embeddings_pos_neg.append(pos_embeddings_user)
                item_embeddings_pos_neg.append(pos_embeddings_item)
            if len(neg_index) > 0:
                user_embeddings_pos_neg.append(neg_embeddings_user)
                item_embeddings_pos_neg.append(neg_embeddings_item)


            # 把pos_neg embeeding拼起来作为最终的embedding
            user_embedding = torch.cat(user_embeddings_pos_neg, dim = 1)
            item_embedding = torch.cat(item_embeddings_pos_neg, dim = 1)
            #print("pos_index, neg_index",pos_index, neg_index)
            #print("final shape", user_embedding.shape)
            #input("确认")

            # user_embedding = torch.cat(user_embeddings_each_feedback, dim = 1)
            # item_embedding = torch.cat(item_embeddings_each_feedback, dim = 1)

            # user_embeddings_each_feedback = torch.stack(user_embeddings_each_feedback, dim = 0)
            # item_embeddings_each_feedback = torch.stack(item_embeddings_each_feedback, dim = 0)

            # user_embeddings_each_feedback = torch.stack(user_embeddings_each_feedback, dim = 0)
            # item_embeddings_each_feedback = torch.stack(item_embeddings_each_feedback, dim = 0)

            # user_embedding = torch.matmul(torch.mean(user_embeddings_each_feedback, dim = 0), self.weight_dic["coarse_weight_u"])
            # item_embedding = torch.matmul(torch.mean(item_embeddings_each_feedback, dim = 0), self.weight_dic["coarse_weight_v"])
        
        elif self.combine_choice == 'cat':
            EP_index = torch.squeeze((self.behavior_type == 0).nonzero(), dim = -1)
            if len(EP_index) > 0:
                EP_embeddings_user = torch.matmul(torch.cat(list(user_embeddings_each_behavior[EP_index]), dim = 1), self.weight_dic["fine_weight_EP_u"])
                EP_embeddings_item = torch.matmul(torch.cat(list(item_embeddings_each_behavior[EP_index]), dim = 1), self.weight_dic["fine_weight_EP_v"])

            IP_index = torch.squeeze((self.behavior_type == 1).nonzero(), dim = -1)
            if len(IP_index) > 0:
                IP_embeddings_user = torch.matmul(torch.cat(list(user_embeddings_each_behavior[IP_index]), dim = 1), self.weight_dic["fine_weight_IP_u"])
                IP_embeddings_item = torch.matmul(torch.cat(list(item_embeddings_each_behavior[IP_index]), dim = 1), self.weight_dic["fine_weight_IP_v"])

            EN_index = torch.squeeze((self.behavior_type == 2).nonzero(), dim = -1)
            if len(EN_index) > 0:
                EN_embeddings_user = torch.matmul(torch.cat(list(user_embeddings_each_behavior[EN_index]), dim = 1), self.weight_dic["fine_weight_EN_u"])
                EN_embeddings_item = torch.matmul(torch.cat(list(item_embeddings_each_behavior[EN_index]), dim = 1), self.weight_dic["fine_weight_EN_v"])

            IN_index = torch.squeeze((self.behavior_type == 3).nonzero(), dim = -1)
            if len(IN_index) > 0:
                IN_embeddings_user = torch.matmul(torch.cat(list(user_embeddings_each_behavior[IN_index]), dim = 1), self.weight_dic["fine_weight_IN_u"])
                IN_embeddings_item = torch.matmul(torch.cat(list(item_embeddings_each_behavior[IN_index]), dim = 1), self.weight_dic["fine_weight_IN_v"])


            # --------------- denoise IP and IN by orthogonal mapping --------------- 
            if self.denoise_flag:
                # denoise IP with EN
                if len(IP_index) > 0 and len(EN_index) > 0:
                    IP_embeddings_user = self.orthogonal_denoise(IP_embeddings_user, EN_embeddings_user)
                    IP_embeddings_item = self.orthogonal_denoise(IP_embeddings_item, EN_embeddings_item)
                # denoise IN with EP
                if len(IN_index) > 0 and len(EP_index) > 0:
                    IN_embeddings_user = self.orthogonal_denoise(IN_embeddings_user, EP_embeddings_user)
                    IN_embeddings_item = self.orthogonal_denoise(IN_embeddings_item, EP_embeddings_item)

            if len(EP_index) > 0: 
                user_embeddings_each_feedback.append(EP_embeddings_user)
                item_embeddings_each_feedback.append(EP_embeddings_item)
            
            if len(IP_index) > 0:
                user_embeddings_each_feedback.append(IP_embeddings_user)
                item_embeddings_each_feedback.append(IP_embeddings_item)
                
            if len(EN_index) > 0:
                user_embeddings_each_feedback.append(EN_embeddings_user)
                item_embeddings_each_feedback.append(EN_embeddings_item)
                
            if len(IN_index) > 0:
                user_embeddings_each_feedback.append(IN_embeddings_user)
                item_embeddings_each_feedback.append(IN_embeddings_item)
        
            

            # --------------- combine feedback embeddings to form the final user and item embedding --------------- 
            # user_embedding = torch.matmul(torch.cat(user_embeddings_each_feedback, dim = 1), self.weight_dic["coarse_weight_u"])
            # item_embedding = torch.matmul(torch.cat(item_embeddings_each_feedback, dim = 1), self.weight_dic["coarse_weight_v"])

            user_embedding = torch.cat(user_embeddings_each_feedback, dim = 1)
            item_embedding = torch.cat(item_embeddings_each_feedback, dim = 1)

            user_embeddings_each_feedback = torch.stack(user_embeddings_each_feedback, dim = 0)
            item_embeddings_each_feedback = torch.stack(item_embeddings_each_feedback, dim = 0)
        

        kinds_of_embeddings = {}
        kinds_of_embeddings['user_embeddings_each_behavior'] = user_embeddings_each_behavior
        kinds_of_embeddings['item_embeddings_each_behavior'] = item_embeddings_each_behavior
        kinds_of_embeddings['user_embeddings_each_feedback'] = user_embeddings_each_feedback
        kinds_of_embeddings['item_embeddings_each_feedback'] = item_embeddings_each_feedback
        kinds_of_embeddings['user_embedding'] = user_embedding
        kinds_of_embeddings['item_embedding'] = item_embedding

        # for key in kinds_of_embeddings:
        #     if torch.isnan(kinds_of_embeddings[key]).any():
        #         print(key)
        #         print("is nan")
        #         input("pause")
            
        #     if torch.isinf(kinds_of_embeddings[key]).any():
        #         print(key)
        #         print("is inf")
        #         input("pause")


        return kinds_of_embeddings


    def orthogonal_denoise(self, emb1,emb2):
        """
        denoise emb1 with emb2: emb1 - org_map emb1 onto emb2
        """
        temp1 = torch.sum(emb1 * emb2, dim = 1)
        temp2 = torch.sum(emb2 * emb2, dim = 1) + 1e-20 
        temp3 = (temp1 / temp2).reshape((-1, 1))
        # if torch.isnan(temp3).any():
        #     print("nan in temp3")
        #     input("pause")
        # if torch.isinf(temp3).any():
        #     print("inf in temp3")
        #     input("pause")

        noise = temp3 * emb2
        res = emb1 - noise

        return res    




class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, n_layers, sparse_graph_flag, u2u_flag, i2i_flag, batch_norm, dropout):
        """
        params:
            num_users: number of users
            num_items: number of items
            embed_dim: dimension of embedding
            n_layers: number of GCN layers
        """
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.user_embedding, self.item_embedding = self.init_embedding()   
        self.sparse_graph_flag = sparse_graph_flag
        self.u2u_flag = u2u_flag
        self.i2i_flag = i2i_flag
        self.batch_norm = batch_norm
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(LightGCN_Layer(self.sparse_graph_flag, self.u2u_flag, self.i2i_flag, self.batch_norm, self.embed_dim))


    def init_embedding(self):
        user_embedding = nn.Embedding(self.num_users, self.embed_dim)
        item_embedding = nn.Embedding(self.num_items, self.embed_dim)
        nn.init.xavier_uniform_(user_embedding.weight)
        nn.init.xavier_uniform_(item_embedding.weight)

        return user_embedding, item_embedding


    def forward(self, u2u_adj, beh_u2i_fold_list, beh_i2u_fold_list, beh_user_item_index, i2i_adj, mask_flag, device):
        """
        param:
        
        return:
            the learned embeddings of users and items
        """
        device = torch.device(device)

        # 要检查是否各层不一样
        user_embeddings_each_layer = []
        item_embeddings_each_layer = []

        user_embedding = self.user_embedding.weight
        item_embedding = self.item_embedding.weight

        user_embeddings_each_layer.append(user_embedding)
        item_embeddings_each_layer.append(item_embedding)
    
        for i in range(self.n_layers):
            user_embedding, item_embedding = self.layers[i](user_embedding, item_embedding, u2u_adj, beh_u2i_fold_list, beh_i2u_fold_list, i2i_adj)
            user_embedding = self.dropout(user_embedding)
            item_embedding = self.dropout(item_embedding)

            user_embeddings_each_layer.append(user_embedding)
            item_embeddings_each_layer.append(item_embedding)

        user_embeddings_each_layer = torch.stack(user_embeddings_each_layer, dim = 0)
        item_embeddings_each_layer = torch.stack(item_embeddings_each_layer, dim = 0)

        user_embedding = torch.mean(user_embeddings_each_layer, dim = 0)
        item_embedding = torch.mean(item_embeddings_each_layer, dim = 0)


        if mask_flag:
            mask_mat_user = torch.zeros((user_embedding.shape[0],1)).to(device)
            mask_mat_item = torch.zeros((item_embedding.shape[0],1)).to(device)

            beh_user_index = beh_user_item_index[0]
            beh_item_index = beh_user_item_index[1]
            
            mask_mat_user[beh_user_index.to(torch.long)] = 1
            mask_mat_item[beh_item_index.to(torch.long)] = 1

            user_embedding = user_embedding * mask_mat_user
            item_embedding = item_embedding * mask_mat_item
            
            
        return user_embedding, item_embedding
        
    



class LightGCN_Layer(nn.Module):
    def __init__(self, sparse_graph_flag, u2u_flag, i2i_flag, batch_norm, embed_dim):
        super(LightGCN_Layer, self).__init__()
        self.sparse_graph_flag = sparse_graph_flag
        self.u2u_flag = u2u_flag
        self.i2i_flag = i2i_flag
        self.batch_norm = batch_norm
        self.bn = nn.BatchNorm1d(embed_dim) if batch_norm else None

    def forward(self, user_embedding, item_embedding, u2u_adj, beh_u2i_fold_list, beh_i2u_fold_list, i2i_adj):
        """
        param:
            user_embedding: user embedding from last layer   
            item_embedding: item embedding from last layer
    
        return:
            the learned embeddings from this layer
        """
        if self.sparse_graph_flag:
            # user embedding aggregation
            if self.u2u_flag:
                u2u_agg = torch.spmm(u2u_adj, user_embedding)

            i2u_aggs_list = []
            for u2i_fold in beh_u2i_fold_list:
                i2u_aggs_list.append(self.my_spmm(u2i_fold, item_embedding))
            i2u_agg = torch.cat(i2u_aggs_list, dim = 0)
                    

            
            # print("ok i2u")
            # i2u_agg = torch.spmm(u2i_adj, item_embedding)

            
            # u2i_indices = u2i_adj._indices()
            # u2i_values = u2i_adj._values()
            # row = u2i_adj.size()[0]
            # col = u2i_adj.size()[1]
            # i2u_agg = torch_sparse.spmm(u2i_indices, u2i_values, row, col, item_embedding)

            #item embedding aggregation
            if self.i2i_flag:
                i2i_agg = torch.spmm(i2i_adj, item_embedding)
            # u2i_agg = torch.spmm(u2i_adj.transpose(-1,-2), user_embedding)
            u2i_aggs_list = []
            for i2u_fold in beh_i2u_fold_list:
                u2i_aggs_list.append(self.my_spmm(i2u_fold, user_embedding))
            u2i_agg = torch.cat(u2i_aggs_list, dim = 0)
            
            # print("ok u2i")
            # i2u_adj = u2i_adj.transpose(-1,-2)
            # i2u_indices = i2u_adj._indices()
            # i2u_values = i2u_adj._values()
            # row = i2u_adj.size()[0]
            # col = i2u_adj.size()[1]
            # u2i_agg = torch_sparse.spmm(i2u_indices, i2u_values, row, col, user_embedding)

        # else:
        #     # user embedding aggregation
        #     if self.u2u_flag:
        #         u2u_agg = torch.matmul(u2u_adj, user_embedding)
        #     i2u_agg = torch.matmul(u2i_adj, item_embedding)

        #     # item embedding aggregation
        #     if self.i2i_flag:
        #         i2i_agg = torch.matmul(i2i_adj, item_embedding)
        #     u2i_agg = torch.matmul(u2i_adj.transpose(-1,-2), user_embedding)


        # aggregate message for users and items (mean / concatnate / max /)
        if self.u2u_flag:
            # user_embedding = F.relu((u2u_agg + i2u_agg)/2.0)
            user_embedding = u2u_agg + i2u_agg
        else:
            user_embedding = i2u_agg
        
        if self.i2i_flag:
            # item_embedding = F.relu((i2i_agg + u2i_agg)/2.0)
            item_embedding = i2i_agg + u2i_agg
        else:
            item_embedding = u2i_agg

        if self.batch_norm:
            user_embedding = self.bn(user_embedding)
            item_embedding = self.bn(item_embedding)
            #user_embedding = F.normalize(user_embedding, p=2, dim=0)
            #item_embedding = F.normalize(item_embedding, p=2, dim=0)

        return user_embedding, item_embedding
        

    def my_spmm(self, sp_mat, dens_mat):
        sp_indices = sp_mat._indices()
        sp_values = sp_mat._values()
        row = sp_mat.size()[0]
        col = sp_mat.size()[1]
        agg = torch_sparse.spmm(sp_indices, sp_values, row, col, dens_mat)

        return agg

"""
Multi_LightGCN
    拼接，去噪
    LightGCN
        LightGCN_Layer
"""

