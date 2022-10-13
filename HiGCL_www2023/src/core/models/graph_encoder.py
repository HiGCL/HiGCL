import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch_sparse
import time


class graph_encoder(nn.Module):
    def __init__(self, config, userNum, itemNum):
        super(graph_encoder, self).__init__() 
        self.config = config 
        self.userNum = userNum
        self.itemNum = itemNum
        self.act_func = nn.ReLU()
        self.user_embedding, self.item_embedding = self.init_embedding()        

        self.weight_dic = self.init_weight()

        self.feedback_layers = nn.ModuleList()
        for _ in range(4):   # 0 EP , 1 IP , 2 EN, 3 IN
            curfeedback_layers = nn.ModuleList()
            for _ in range(self.config['n_layers']):  
                curfeedback_layers.append(GCN_Layer(self.config['dim_each_feedback'], self.config['dim_each_feedback']))  
            self.feedback_layers.append(curfeedback_layers)


    def init_embedding(self):
        user_embedding = torch.nn.Embedding(self.userNum, self.config['dim_each_feedback'])
        item_embedding = torch.nn.Embedding(self.itemNum, self.config['dim_each_feedback'])
        nn.init.xavier_uniform_(user_embedding.weight)
        nn.init.xavier_uniform_(item_embedding.weight)

        return user_embedding, item_embedding


    def init_weight(self):
        weight_dic = {}
        for i in range(4):  # EP 0, IP 1 , EN 2, IN 3
            i_concatenation_w = nn.Parameter(torch.Tensor(self.config['n_layers']*self.config['dim_each_feedback'], self.config['dim_each_feedback']))
            u_concatenation_w = nn.Parameter(torch.Tensor(self.config['n_layers']*self.config['dim_each_feedback'], self.config['dim_each_feedback']))
            init.xavier_uniform_(i_concatenation_w)
            init.xavier_uniform_(u_concatenation_w)
            weight_dic[str(i)+"_user"] = u_concatenation_w
            weight_dic[str(i)+"_item"] = i_concatenation_w

        pos_weight_u = nn.Parameter(torch.Tensor(self.config['dim_each_feedback'], self.config['dim_each_feedback']), requires_grad=True)
        neg_weight_u = nn.Parameter(torch.Tensor(self.config['dim_each_feedback'], self.config['dim_each_feedback']), requires_grad=True)
        pos_weight_v = nn.Parameter(torch.Tensor(self.config['dim_each_feedback'], self.config['dim_each_feedback']), requires_grad=True)
        neg_weight_v = nn.Parameter(torch.Tensor(self.config['dim_each_feedback'], self.config['dim_each_feedback']), requires_grad=True)
        init.xavier_uniform_(pos_weight_u)
        init.xavier_uniform_(neg_weight_u)
        init.xavier_uniform_(pos_weight_v)
        init.xavier_uniform_(neg_weight_v)
        weight_dic['pos_weight_u'] = pos_weight_u
        weight_dic['pos_weight_v'] = pos_weight_v
        weight_dic['neg_weight_u'] = neg_weight_u
        weight_dic['neg_weight_v'] = neg_weight_v
        weight_dic = nn.ParameterDict(weight_dic)
        return weight_dic


    def forward(self, multi_adjs):
        user_embeddings_each_behavior = []
        item_embeddings_each_behavior = []
        user_embeddings_each_feedback = []
        item_embeddings_each_feedback = []
        
        for i in range(4): # EP 0, IP 1 , EN 2, IN 3
            cur_feedback_index = torch.squeeze((torch.tensor(self.config['behavior_type']) == i).nonzero(), dim = -1)
            if len(cur_feedback_index) == 0:
                continue
           
            user_embedding = self.user_embedding.weight
            item_embedding = self.item_embedding.weight
    
            all_user_embeddings = []
            all_item_embeddings = []
            all_user_embeddingss = []
            all_item_embeddingss = []

            list_beh_u2i_fold_list = []
            list_beh_i2u_fold_list = []
            for beh_index in cur_feedback_index:
                list_beh_u2i_fold_list.append(multi_adjs['multi_u2i_fold_list'][beh_index])
                list_beh_i2u_fold_list.append(multi_adjs['multi_i2u_fold_list'][beh_index])

            for _, layer in enumerate(self.feedback_layers[i]):
                user_embedding, item_embedding, user_embeddings, item_embeddings = layer(user_embedding, item_embedding, list_beh_u2i_fold_list, list_beh_i2u_fold_list)

                all_user_embeddings.append(user_embedding)
                all_item_embeddings.append(item_embedding)
                all_user_embeddingss.append(user_embeddings)
                all_item_embeddingss.append(item_embeddings)
                
            user_embedding = torch.cat(all_user_embeddings, dim=1)
            item_embedding = torch.cat(all_item_embeddings, dim=1)
            user_embeddings = torch.cat(all_user_embeddingss, dim=2)
            item_embeddings = torch.cat(all_item_embeddingss, dim=2)

            cur_feedback_user_embedding = torch.matmul(user_embedding , self.weight_dic[str(i)+"_user"])      # embedding of current feedback
            cur_feedback_item_embedding = torch.matmul(item_embedding , self.weight_dic[str(i)+"_item"])
            cur_feedback_user_embeddings = torch.matmul(user_embeddings , self.weight_dic[str(i)+"_user"])    # embeddings of kinds of behaviors
            cur_feedback_item_embeddings = torch.matmul(item_embeddings , self.weight_dic[str(i)+"_item"])

            # print("cur_feedback_user_embedding.shape", cur_feedback_user_embedding.shape)
            # print("cur_feedback_item_embedding.shape", cur_feedback_item_embedding.shape)
            # print("cur_feedback_user_embeddings.shape", cur_feedback_user_embeddings.shape)
            # print("cur_feedback_item_embeddings.shape", cur_feedback_item_embeddings.shape)

            for k in range(cur_feedback_user_embeddings.shape[0]):
                user_embeddings_each_behavior.append(cur_feedback_user_embeddings[k])
                item_embeddings_each_behavior.append(cur_feedback_item_embeddings[k])

            user_embeddings_each_feedback.append(cur_feedback_user_embedding)
            item_embeddings_each_feedback.append(cur_feedback_item_embedding)

        user_embeddings_each_behavior = torch.stack(user_embeddings_each_behavior, dim = 0)
        item_embeddings_each_behavior = torch.stack(item_embeddings_each_behavior, dim = 0)
        
        

        # --------------- denoise IP and IN by orthogonal mapping --------------- 
        if self.config['denoise_flag']:
            # denoise IP with EN
            IP_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type']) == 1).nonzero(), dim = -1)
            EN_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type']) == 2).nonzero(), dim = -1)
            if len(IP_index) > 0 and len(EN_index) > 0:
                user_embeddings_each_feedback[IP_index[0]] = self.orthogonal_denoise(user_embeddings_each_feedback[IP_index[0]], user_embeddings_each_feedback[EN_index[0]])
                item_embeddings_each_feedback[IP_index[0]] = self.orthogonal_denoise(item_embeddings_each_feedback[IP_index[0]], item_embeddings_each_feedback[EN_index[0]])

            # denoise IN with EP
            IN_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type']) == 3).nonzero(), dim = -1)
            EP_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type']) == 0).nonzero(), dim = -1)
            if len(IN_index) > 0 and len(EP_index) > 0:
                user_embeddings_each_feedback[IN_index[0]] = self.orthogonal_denoise(user_embeddings_each_feedback[IN_index[0]], user_embeddings_each_feedback[EP_index[0]])
                item_embeddings_each_feedback[IN_index[0]] = self.orthogonal_denoise(item_embeddings_each_feedback[IN_index[0]], item_embeddings_each_feedback[EP_index[0]])


        # --------------- combine feedback embeddings to form the final user and item embedding --------------- 
        user_embeddings_each_feedback = torch.stack(user_embeddings_each_feedback, dim = 0)
        item_embeddings_each_feedback = torch.stack(item_embeddings_each_feedback, dim = 0)

        # fuse feedback embeddings into positive and negative embedding
        pos_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type']) <= 1).nonzero(), dim = -1)
        neg_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type']) > 1).nonzero(), dim = -1)


        if len(pos_index) > 0:
            #pos_embeddings_user = self.act_func(torch.matmul(torch.mean(user_embeddings_each_feedback[pos_index], dim = 0), self.weight_dic["pos_weight_u"]))
            #pos_embeddings_item = self.act_func(torch.matmul(torch.mean(item_embeddings_each_feedback[pos_index], dim = 0), self.weight_dic["pos_weight_v"]))
            pos_embeddings_user = torch.mean(user_embeddings_each_feedback[pos_index], dim = 0)
            pos_embeddings_item = torch.mean(item_embeddings_each_feedback[pos_index], dim = 0)

        if len(neg_index) > 0:
            # neg_embeddings_user = self.act_func(torch.matmul(torch.mean(user_embeddings_each_feedback[neg_index], dim = 0), self.weight_dic["neg_weight_u"]))
            # neg_embeddings_item = self.act_func(torch.matmul(torch.mean(item_embeddings_each_feedback[neg_index], dim = 0), self.weight_dic["neg_weight_v"]))
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


        user_embedding = torch.cat(user_embeddings_pos_neg, dim = 1)
        item_embedding = torch.cat(item_embeddings_pos_neg, dim = 1)

        # print("user_embeddings_each_behavior.shape", user_embeddings_each_behavior.shape)
        # print("item_embeddings_each_behavior.shape", item_embeddings_each_behavior.shape)
        # print("user_embeddings_each_feedback.shape", user_embeddings_each_feedback.shape)
        # print("item_embeddings_each_feedback.shape", item_embeddings_each_feedback.shape)

        kinds_of_embeddings = {}
        kinds_of_embeddings['user_embeddings_each_behavior'] = user_embeddings_each_behavior
        kinds_of_embeddings['item_embeddings_each_behavior'] = item_embeddings_each_behavior
        kinds_of_embeddings['user_embeddings_each_feedback'] = user_embeddings_each_feedback
        kinds_of_embeddings['item_embeddings_each_feedback'] = item_embeddings_each_feedback
        kinds_of_embeddings['user_embedding'] = user_embedding
        kinds_of_embeddings['item_embedding'] = item_embedding


        return kinds_of_embeddings


    def orthogonal_denoise(self, emb1,emb2):
        """
        denoise emb1 with emb2: emb1 - org_map emb1 onto emb2
        """
        temp1 = torch.sum(emb1 * emb2, dim = 1)
        temp2 = torch.sum(emb2 * emb2, dim = 1) + 1e-20 
        temp3 = (temp1 / temp2).reshape((-1, 1))

        noise = temp3 * emb2
        res = emb1 - noise

        return res    




class GCN_Layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN_Layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.act = torch.nn.Sigmoid()
        self.weight_dic = self.init_weight()


    def init_weight(self):
        weight_dic = {}
        i_w = nn.Parameter(torch.Tensor(self.in_dim, self.out_dim))
        u_w = nn.Parameter(torch.Tensor(self.in_dim, self.out_dim))
        init.xavier_uniform_(i_w)
        init.xavier_uniform_(u_w)
        weight_dic['i_w'] = i_w
        weight_dic['u_w'] = u_w
        weight_dic = nn.ParameterDict(weight_dic)
        return weight_dic


    def forward(self, user_embedding, item_embedding, list_beh_u2i_fold_list, list_beh_i2u_fold_list):

        num_cur_feedback_beh = len(list_beh_i2u_fold_list)    
        user_embedding_list = [None]*num_cur_feedback_beh
        item_embedding_list = [None]*num_cur_feedback_beh
        orig_user_embedding = user_embedding
        orig_item_embedding = item_embedding


        for i in range(num_cur_feedback_beh):
            beh_u2i_fold_list = list_beh_u2i_fold_list[i]
            i2u_aggs_list = []
            for u2i_fold in beh_u2i_fold_list:
                i2u_aggs_list.append(torch.spmm(u2i_fold, orig_item_embedding))
            i2u_agg = torch.cat(i2u_aggs_list, dim = 0)
            
            beh_i2u_fold_list = list_beh_i2u_fold_list[i]
            u2i_aggs_list = []
            for i2u_fold in beh_i2u_fold_list:
                u2i_aggs_list.append(torch.spmm(i2u_fold, orig_user_embedding))
            u2i_agg = torch.cat(u2i_aggs_list, dim = 0) 
            
            user_embedding_list[i] = i2u_agg
            item_embedding_list[i] = u2i_agg

        user_embeddings = torch.stack(user_embedding_list, dim=0) 
        item_embeddings = torch.stack(item_embedding_list, dim=0)
        
        user_embedding = self.act(torch.matmul(torch.mean(user_embeddings, dim=0), self.weight_dic['u_w']))   
        item_embedding = self.act(torch.matmul(torch.mean(item_embeddings, dim=0), self.weight_dic['i_w']))

        user_embeddings = self.act(torch.matmul(user_embeddings, self.weight_dic['u_w']))
        item_embeddings = self.act(torch.matmul(item_embeddings, self.weight_dic['i_w']))      

 
        return user_embedding, item_embedding, user_embeddings, item_embeddings    
        


    def my_spmm(self, sp_mat, dens_mat):
        sp_indices = sp_mat._indices()
        sp_values = sp_mat._values()
        row = sp_mat.size()[0]
        col = sp_mat.size()[1]
        agg = torch_sparse.spmm(sp_indices, sp_values, row, col, dens_mat)

        return agg

