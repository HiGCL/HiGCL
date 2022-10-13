import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm


class GraphLearner(nn.Module):
    def __init__(self, input_size, hidden_size, coe_lambda, sparse_graph_flag, topk_u2u=None, topk_i2i=None, \
        epsilon_u2u=None, epsilon_i2i=None, num_pers=2, metric_type='attention'):
        """
        params:
            input_size: input embedding size
            hidden_size: size of the output feature by attention transformation matrix
            coe_lambda: coefficient lambda to balance initial adj with newly learned one
            topk: hyperparameter to construct topk similar nodes as neighbors
            epsilon: similarity score larger than epsilon as neighbors
            num_pers: number of perspective (like multi-head)
            metric_type: similarity metric
            whether_u2i: whether to update the links between user and item during the graph learning process
            graph_include_self: whether self-connected
        """
        super(GraphLearner, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.coe_lambda = coe_lambda
        self.topk_u2u = topk_u2u 
        self.topk_i2i = topk_i2i
        self.epsilon_u2u = epsilon_u2u
        self.epsilon_i2i = epsilon_i2i
        self.num_pers = num_pers
        self.metric_type = metric_type
        self.sparse_graph_flag = sparse_graph_flag

        if self.metric_type == 'attention':
            self.linear_sims_user = nn.ModuleList([nn.Linear(self.input_size, self.hidden_size, bias=False) for _ in range(self.num_pers)])
            self.linear_sims_item = nn.ModuleList([nn.Linear(self.input_size, self.hidden_size, bias=False) for _ in range(self.num_pers)])
            print('[ Multi-perspective {} GraphLearner: {} ]'.format(self.metric_type, self.num_pers))

        elif self.metric_type == 'weighted_cosine':
            self.weight_tensor_user = torch.Tensor(self.num_pers, self.input_size)
            self.weight_tensor_user = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor_user))
            self.weight_tensor_item = torch.Tensor(self.num_pers, self.input_size)
            self.weight_tensor_item = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor_item))
            print('[ Multi-perspective {} GraphLearner: {} ]'.format(self.metric_type, self.num_pers))

        else:
            raise ValueError('Unknown metric_type: {}'.format(self.metric_type))

        print('[ Graph Learner metric type: {} ]'.format(self.metric_type))


    def forward(self, init_adjs, user_embedding, item_embedding):
        """
        params:
            init_adjs: dic_type, u2u_adj, i2i_adj, multi_u2i_adj (list)
            user_embedding: current user embedding
            item_embedding: current item embedding

        return:
            learned graph 
        """

        # --------------- generate new similarity matrix --------------- 
        sim_u2u = 0
        sim_i2i = 0

        if self.metric_type == 'attention':
            for _ in range(len(self.linear_sims_user)):
                attention_embed_user = torch.relu(self.linear_sims_user[_](user_embedding))
                sim_u2u += torch.matmul(attention_embed_user, attention_embed_user.transpose(-1, -2))
                attention_embed_item = torch.relu(self.linear_sims_item[_](item_embedding))
                sim_i2i += torch.matmul(attention_embed_item, attention_embed_item.transpose(-1, -2))
        
            sim_u2u /= len(self.linear_sims_user)
            sim_i2i /= len(self.linear_sims_item)

        elif self.metric_type == 'weighted_cosine':     
            expand_weight_tensor_user = self.weight_tensor_user.unsqueeze(1)
            weighted_embedding_user = user_embedding.unsqueeze(0) * expand_weight_tensor_user
            user_norm = F.normalize(weighted_embedding_user, p=2, dim=-1)
            sim_u2u = torch.matmul(user_norm, user_norm.transpose(-1, -2)).mean(0)
            
            expand_weight_tensor_item = self.weight_tensor_item.unsqueeze(1)
            weighted_embedding_item = item_embedding.unsqueeze(0) * expand_weight_tensor_item
            item_norm = F.normalize(weighted_embedding_item, p=2, dim=-1)
            sim_i2i = torch.matmul(item_norm, item_norm.transpose(-1, -2)).mean(0)

        markoff_value = 0

        
        # --------------- combine with init adjacency matrix to generate new ones  ----------------

        # sim_u2u_adj = torch.softmax(sim_u2u_adj, dim=-1)
        # sim_i2i_adj = torch.softmax(sim_i2i_adj, dim=-1)

        if self.metric_type == 'attention':
            sim_u2u_adj = torch.sigmoid(sim_u2u_adj)
            sim_i2i_adj = torch.sigmoid(sim_i2i_adj)
        else:
            sim_u2u_adj = sim_u2u
            sim_i2i_adj = sim_i2i
        
        if self.epsilon_u2u is not None:
            sim_u2u_adj, sim_i2i_adj, = \
                self.build_epsilon_neighbourhood(self.epsilon_u2u, self.epsilon_i2i, markoff_value, sim_u2u_adj, sim_i2i_adj)

        if self.topk_u2u is not None:
            sim_u2u_adj, sim_i2i_adj = \
                self.build_knn_neighbourhood(self, self.topk_u2u, self.topk_i2i, markoff_value, sim_u2u_adj, sim_i2i_adj)
    

        u2u_adj = self.coe_lambda * init_adjs['u2u_adj'] + (1 - self.coe_lambda) * sim_u2u_adj
        i2i_adj = self.coe_lambda * init_adjs['i2i_adj'] + (1 - self.coe_lambda) * sim_i2i_adj


        new_adjs = {}
        new_adjs['u2u_adj'] = u2u_adj
        new_adjs['i2i_adj'] = i2i_adj
        new_adjs['multi_u2i_adj'] = init_adjs['multi_u2i_adj']

        return new_adjs



    def build_knn_neighbourhood(self, topk_u2u, topk_i2i, markoff_value, sim_u2u, sim_i2i):
        topk_u2u = min(topk_u2u, sim_u2u.size(-1))
        knn_val_u2u, knn_ind_u2u = torch.topk(sim_u2u, topk_u2u, dim=-1)
        u2u_adj = to_cuda((markoff_value * torch.ones_like(sim_u2u)).scatter_(-1, knn_ind_u2u, knn_val_u2u), self.device)

        topk_i2i = min(topk_i2i, sim_i2i.size(-1))
        knn_val_i2i, knn_ind_i2i = torch.topk(sim_i2i, topk_i2i, dim=-1)
        i2i_adj = to_cuda((markoff_value * torch.ones_like(sim_i2i)).scatter_(-1, knn_ind_i2i, knn_val_i2i), self.device)
        
        if self.sparse_graph_flag:
            # convert dense matrix to torch.sparse matrix
            u2u_index = (u2u_adj != markoff_value).nonzero().reshape((2,-1))
            u2u_val = u2u_adj[u2u_index[0,:], u2u_index[1,:]]
            u2u_shape = u2u_adj.shape
            u2u_adj = torch.sparse_coo_tensor(u2u_index, u2u_val, u2u_shape).to(torch.float32)

            i2i_index = (i2i_adj != markoff_value).nonzero().reshape((2,-1))
            i2i_val = i2i_adj[i2i_index[0,:], i2i_index[1,:]]
            i2i_shape = i2i_adj.shape
            i2i_adj = torch.sparse_coo_tensor(i2i_index, i2i_val, i2i_shape).to(torch.float32)
    
        return u2u_adj, i2i_adj



    def build_epsilon_neighbourhood(self, epsilon_u2u, epsilon_i2i, markoff_value, sim_u2u, sim_i2i):
        mask_u2u = (sim_u2u > epsilon_u2u).detach().float()
        u2u_adj = sim_u2u * mask_u2u + markoff_value * (1 - mask_u2u)

        mask_i2i = (sim_i2i > epsilon_i2i).detach().float()
        i2i_adj = sim_i2i * mask_i2i + markoff_value * (1 - mask_i2i)

        if self.sparse_graph_flag:
            # convert dense matrix to torch.sparse matrix
            u2u_index = (u2u_adj != markoff_value).nonzero().reshape((2,-1))
            u2u_val = u2u_adj[u2u_index[0,:], u2u_index[1,:]]
            u2u_shape = u2u_adj.shape
            u2u_adj = torch.sparse_coo_tensor(u2u_index, u2u_val, u2u_shape).to(torch.float32)

            i2i_index = (i2i_adj != markoff_value).nonzero().reshape((2,-1))
            i2i_val = i2i_adj[i2i_index[0,:], i2i_index[1,:]]
            i2i_shape = i2i_adj.shape
            i2i_adj = torch.sparse_coo_tensor(i2i_index, i2i_val, i2i_shape).to(torch.float32)

        return u2u_adj, i2i_adj

    
