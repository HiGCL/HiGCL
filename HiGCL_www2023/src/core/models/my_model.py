from operator import ne
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mul_beh_encoder import Multi_Encoder
from .new_encoder import new_Multi_Encoder
from .feedback_encoder import Feedback_Encoder
from .my_graphlearn import GraphLearner
import torch.optim as optim
from .graph_encoder import graph_encoder

class My_Model(nn.Module):
    """
    overall structure of the proposed model
    Multi-encoder + Graph Learner 
    """
    def __init__(self, config, init_adj, num_users, num_items):
        super(My_Model, self).__init__()
        # prepare parameters
        # overall
        self.config = config
        self.name = 'My_Model'
        self.dim_each_feedback = config['dim_each_feedback']
        self.num_users = num_users
        self.num_items = num_items
        self.sparse_graph_flag = config['sparse_graph_flag']
       
        # graph learning
        self.graph_learn = config['graph_learn']
        self.hidden_size = config['attention_hidden_size']
        self.coe_lambda = config['coe_lambda']
        self.topk_u2u = config['topk_u2u']
        self.topk_i2i = config['topk_i2i']
        self.epsilon_u2u = config['epsilon_u2u']
        self.epsilon_i2i = config['epsilon_i2i']
        self.num_pers = config['num_pers']
        self.metric_type = config['metric_type']
        self.graph_include_self = config['graph_include_self']
        
        # multi_encoder
        self.graph_module = config['graph_module']   
        self.n_layers = config['n_layers']
        self.behavior_names = config['behavior_name']
        self.behavior_type = config['behavior_type']
        self.u2u_flag = config['u2u_flag']
        self.i2i_flag = config['i2i_flag']
        self.batch_norm = config['batch_norm']
        self.dropout = config['dropout']
        self.denoise_flag = config['denoise_flag']

        self.latest_adj = init_adj

        # encoder 
        self.encoder = graph_encoder(self.config, self.num_users, self.num_items)

        # if config['feedback_encoder']:
        #     self.encoder = Feedback_Encoder(self.num_users, self.num_items, self.dim_each_feedback, \
        #             self.n_layers, self.behavior_names, self.behavior_type, self.sparse_graph_flag, combine_choice=config['combine_choice'], graph_module=self.graph_module,\
        #                 u2u_flag = self.u2u_flag, i2i_flag = self.i2i_flag, batch_norm = self.batch_norm, dropout = self.dropout,
        #                 denoise_flag=self.denoise_flag)
        # else:
        #     if config['multi_lightgcn']:
        #         self.encoder = Multi_Encoder(self.num_users, self.num_items, self.dim_each_feedback, \
        #             self.n_layers, self.behavior_names, self.behavior_type, self.sparse_graph_flag, combine_choice=config['combine_choice'], graph_module=self.graph_module,\
        #                 u2u_flag = self.u2u_flag, i2i_flag = self.i2i_flag, batch_norm = self.batch_norm, dropout = self.dropout,
        #                 denoise_flag=self.denoise_flag)
        #     else:
        #         self.encoder = new_Multi_Encoder(self.config, self.config['mlp_flag'],self.config['device'], self.config['mask_flag'], self.num_users, self.num_items, self.dim_each_feedback, \
        #         self.n_layers, self.behavior_names, self.behavior_type, self.sparse_graph_flag, combine_choice=config['combine_choice'], graph_module=self.graph_module,\
        #         u2u_flag = self.u2u_flag, i2i_flag = self.i2i_flag, batch_norm = self.batch_norm, dropout = self.dropout,
        #         denoise_flag=self.denoise_flag)


        # graph learner
        self.num_coarse = len(set(self.behavior_type))
        if self.graph_learn:
            self.graph_learner = GraphLearner(self.num_coarse * self.dim_each_feedback, self.hidden_size, self.coe_lambda, self.sparse_graph_flag, \
                self.topk_u2u, self.topk_i2i,\
        self.epsilon_u2u, self.epsilon_i2i, \
            self.num_pers, self.metric_type, self.graph_include_self)

            print('[ Graph Learner ]')
            if config['graph_learn_regularization']:
                print('[ Graph Regularization]')
        else:
            self.graph_learner = None
            

    def forward(self, user_embedding, item_embedding, init_adjs):
        """
        params:
            user_embedding / item embedding: given / pre-trained embedding or current embedding learned by encoder 
            init_adjs: dic_type, u2u_adj, i2i_adj, multi_u2i_adj (list)

        return:
            new_adj_matrixs: dic type. new adjacency matrix (u2u_adj, i2i_adj, multi_u2i_adj)
            kinds_of_embeddings: user_embeddings_each_behavior, item_embeddings_each_behavior, 
                                 user_embeddings_each_feedback, item_embeddings_each_feedback,
                                 user_embedding, item_embedding
        """
        # learn graph
        if self.graph_learn:
            new_adjs = self.graph_learner(init_adjs, user_embedding, item_embedding)
        else:
            new_adjs = init_adjs

        # using updated adjacency matrixs to update embeddings
        kinds_of_embeddings = self.encoder(init_adjs)

        # self.latest_adj = new_adjs

        return kinds_of_embeddings

