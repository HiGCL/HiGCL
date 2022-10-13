import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .my_model import My_Model
from ..utils import my_DataHandler
from ..utils.time_logger import log
from tqdm import tqdm
import os

class ModelHandler(object):
    """High level model_handler that trains/validates/tests the network,
    tracks and logs metrics.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config['device'])
        self.all_beh_names = {'Taobao': ['purchase', 'cart', 'pageview'],\
            'Beibei': ['purchase', 'cart', 'pageview'],\
            'New_kuaishou': ['like', 'forward', 'follow', 'click', 'not_shortview','hate','shortview', 'unclick']}

        # prepare datasets
        # 1. load datasets
        log("loading dataset")
        multi_beh_data, multi_beh_id, behavior_name, user_item_data_valid, user_item_data_test, sample_valid_mat, sample_test_mat = \
            my_DataHandler.load_data(self.config)
        log("finish loading")

        # 2. create dataset loader
        # train dataset
        train_dataset = my_DataHandler.RecDataset_beh(None, multi_beh_data, \
            multi_beh_id, self.config['behavior_id'], self.config['graph_norm_flag'], self.config['sparse_graph_flag'], self.config['directed_flag'],\
                self.config['graph_include_self'], self.config['u2u_flag'], self.config['i2i_flag'], self.config['p_cnt'], self.config['target_behavior_id'],\
                self.config['num_users'], self.config['num_items'], self.config['folds'])
        self.train_loader = DataLoader(train_dataset, batch_size=self.config['train_batch_size'], shuffle=True, num_workers=8, pin_memory=False)
        
        # valid dataset 
        valid_dataset = my_DataHandler.RecDataset_test(self.train_loader.dataset.num_users, self.train_loader.dataset.num_items, \
            user_item_data_valid, sample_valid_mat)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.config['test_batch_size'], shuffle=True, num_workers=8, pin_memory=False)

        # test dataset
        test_dataset = my_DataHandler.RecDataset_test(self.train_loader.dataset.num_users, self.train_loader.dataset.num_items, \
            user_item_data_test, sample_test_mat)
        self.test_loader = DataLoader(test_dataset, batch_size=self.config['test_batch_size'], shuffle=True, num_workers=8, pin_memory=False)


        # initialize model
        self.model = My_Model(self.config, None, self.train_loader.dataset.num_users, self.train_loader.dataset.num_items)
        self.model.to(self.device)

        ###########opt##########
        self.opt = torch.optim.AdamW(self.model.parameters(), lr = self.config['learning_rate'], weight_decay = self.config['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.opt, self.config['opt_base_lr'], self.config['opt_max_lr'], step_size_up=5, step_size_down=10, mode='triangular', gamma=0.99, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
        
        for name, p in self.model.named_parameters():
            print(name)
            print("-------------------")

        print(self.model)


        # pre calculate IDCG for ndcg 
        self.cal_pre_IDCG(1000)


    def train(self):
        """
        train model
        """
        iter_per_batch = self.config['iter_per_batch']
        num_epochs = self.config['num_epochs']
      
        for key in self.train_loader.dataset.multi_adjs:
            if self.train_loader.dataset.multi_adjs[key] != None:
                for index in range(len(self.train_loader.dataset.multi_adjs[key])):
                    for j in range(len(self.train_loader.dataset.multi_adjs[key][index])):
                        self.train_loader.dataset.multi_adjs[key][index][j] = self.train_loader.dataset.multi_adjs[key][index][j].to(self.device)

        init_adjs = self.train_loader.dataset.multi_adjs

        self.best_hr_test = np.zeros(len(self.config['topk']))
        self.best_ndcg_test = np.zeros(len(self.config['topk']))

        self.best_hr_valid = np.zeros(len(self.config['topk']))
        self.best_ndcg_valid = np.zeros(len(self.config['topk']))

        log("Test before epoch:")
        hit_ratio_valid, ndcg_valid = self.valid(init_adjs)
        self.best_hr_valid = np.maximum(self.best_hr_valid, hit_ratio_valid)
        self.best_ndcg_valid = np.maximum(self.best_ndcg_valid, ndcg_valid)

        hit_ratio_test, ndcg_test = self.test(init_adjs)
        self.best_hr_test = np.maximum(self.best_hr_test, hit_ratio_test)
        self.best_ndcg_test = np.maximum(self.best_ndcg_test, ndcg_test)

        patience_cnt = 0
        print("---------------------------------------\n\n")     
        
        best_hit_ratio_valid_10 = -1
        best_hit_ratio_test_10 = -1
        best_ndcg_valid_10 = -1
        best_ndcg_test_10 = -1
        hit_ratio_valid_50 = None
        ndcg_valid_50 = None
        hit_ratio_test_50 = None
        ndcg_test_50 = None

        best_hit_ratio_valid_50 = None
        best_ndcg_valid_50 = None
        best_hit_ratio_test_50 = None
        best_ndcg_test_50 = None
        
        best_hit_ratio_valid_50_epoch = None
        best_ndcg_valid_50_epoch = None
        best_hit_ratio_test_50_epoch = None
        best_ndcg_test_50_epoch = None

        best_state_dict = None
        early = False
        # ----------------------------- start epoch ----------------------------------
        for epoch_index in range(num_epochs):
            # update model each batch
            log("epoch {} begein!".format(epoch_index))
            if not os.path.exists(self.config['ng_sample_dir'] + 'multi_bpr_ng_sample_target_'+str(epoch_index)+'.pt'):
                self.train_loader.dataset.ng_sample()
                torch.save(self.train_loader.dataset.sample_pair.clone(), self.config['ng_sample_dir'] + 'multi_bpr_ng_sample_target_'+str(epoch_index)+'.pt')
            else:
                self.train_loader.dataset.sample_pair = torch.load(self.config['ng_sample_dir'] + 'multi_bpr_ng_sample_target_'+str(epoch_index)+'.pt')
    
            epoch_loss_bpr = 0
            epoch_loss_infonce = 0
            epoch_loss_reg = 0
            epoch_loss = 0 

            for user_id, observed_item_id, unobserved_item_id in self.train_loader: 

                user_id = user_id.to(self.device)
                observed_item_id = observed_item_id.to(self.device)
                unobserved_item_id = unobserved_item_id.to(self.device)                
                
                loss = 0.

                for iter_index in range(iter_per_batch):
                    self.model.train()

                    kinds_of_embeddings = self.model(None, None, init_adjs)
                    
                    # BPR loss 
                    if self.config['multi_bpr']:
                        BPR_loss = self.cal_BPR_loss_target_aux(user_id, observed_item_id, unobserved_item_id, kinds_of_embeddings['user_embedding'], kinds_of_embeddings['item_embedding'], kinds_of_embeddings['user_embeddings_each_behavior'], kinds_of_embeddings['item_embeddings_each_behavior'])
        
                    else:
                        users_id_target = user_id[:, self.config['ng_target_behavior_index']]               
                        pos_items_id_target = observed_item_id[:, self.config['ng_target_behavior_index']]
                        neg_items_id_target = unobserved_item_id[:, self.config['ng_target_behavior_index']]
                        BPR_loss = self.cal_BPR_loss_target(users_id_target, pos_items_id_target, neg_items_id_target, kinds_of_embeddings['user_embedding'], kinds_of_embeddings['item_embedding'])


                    # InfoNCE loss
                    if self.config['info_loss_choice'] == "paper":
                        InfoNCE_loss = self.cal_InfoNCE_loss_paper(user_id[:,self.config['ng_target_behavior_index']], observed_item_id[:,self.config['ng_target_behavior_index']], None, kinds_of_embeddings)
                    elif self.config['info_loss_choice'] == "target":
                        InfoNCE_loss = self.cal_InfoNCE_loss_target(user_id, observed_item_id, unobserved_item_id, kinds_of_embeddings).to(self.device)
                    elif self.config['info_loss_choice'] == "one2one":
                        InfoNCE_loss = self.cal_InfoNCE_loss_one2one(user_id, observed_item_id, unobserved_item_id, kinds_of_embeddings).to(self.device)
                    elif self.config['info_loss_choice'] == "NN_PP":
                        InfoNCE_loss = self.cal_InfoNCE_loss_NN_PP(user_id, observed_item_id, unobserved_item_id, kinds_of_embeddings).to(self.device)
                    

                    # regLoss
                    used_index = torch.squeeze((observed_item_id[:,self.config['ng_target_behavior_index']] != -1).nonzero(), dim = -1)
                    cal_user_embedding = kinds_of_embeddings['user_embedding']
                    cal_item_embedding = kinds_of_embeddings['item_embedding']
                    userEmbed = cal_user_embedding[user_id[used_index.to(torch.long), self.config['ng_target_behavior_index']].to(torch.long)]
                    posEmbed = cal_item_embedding[observed_item_id[used_index.to(torch.long), self.config['ng_target_behavior_index']].to(torch.long)]
                    negEmbed = cal_item_embedding[unobserved_item_id[used_index.to(torch.long), self.config['ng_target_behavior_index']].to(torch.long)]
                    regLoss = (torch.norm(userEmbed) ** 2 + torch.norm(posEmbed) ** 2 + torch.norm(negEmbed) ** 2)


                    loss += (BPR_loss + self.config['infonce_loss_coe']*InfoNCE_loss + self.config['regloss_coe']*regLoss)/self.config['train_batch_size']


                epoch_loss_bpr += BPR_loss.item()
                epoch_loss_infonce += InfoNCE_loss.item()
                epoch_loss_reg += regLoss.item()
                # BP and update model parameters
                epoch_loss += loss.item()
                self.opt.zero_grad(set_to_none=True)
                # log("Batch loss {} -- BPR_loss:{}, InfoNCE_loss: {}, reg_loss: {},  epoch_loss: {}".format(epoch_index, BPR_loss.item(), InfoNCE_loss.item(), regLoss.item(), loss.item()))
                
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
                self.opt.step()

            log("Traing Epoch {} -- BPR_loss:{}, InfoNCE_loss: {}, reg_loss: {},  epoch_loss: {}".format(epoch_index, epoch_loss_bpr, epoch_loss_infonce, epoch_loss_reg, epoch_loss))
            

            if epoch_index != 0 and epoch_index%50 == 0:
                torch.save(best_state_dict, "./saved_model/"+str(epoch_index)+"_"+self.config['exp_name']+"_model_parameter_final.pt")
            
            if epoch_index%1 == 0:
                hit_ratio_valid, ndcg_valid = self.valid(init_adjs)
                if hit_ratio_valid[0] > self.best_hr_valid[0]:
                    best_hit_ratio_valid_10 = epoch_index 
                if ndcg_valid[0] > self.best_ndcg_valid[0]:
                    best_ndcg_valid_10 = epoch_index
                      
              
                if hit_ratio_valid[0] < self.best_hr_valid[0] and ndcg_valid[0] < self.best_ndcg_valid[0]:
                    patience_cnt += 1
                else:   
                    best_state_dict = self.model.state_dict()
                    patience_cnt = 0
                    self.best_hr_valid = np.maximum(self.best_hr_valid, hit_ratio_valid)
                    self.best_ndcg_valid = np.maximum(self.best_ndcg_valid, ndcg_valid)
                log("best HR_valid: {}, best_NDCG_valid: {}, best_hit_valid_10: {}, best_ndcg_valid_10: {} \n".format(self.best_hr_valid, self.best_ndcg_valid, best_hit_ratio_valid_10, best_ndcg_valid_10))

                if patience_cnt == self.config['patience']:
                    log("Early stop at {} :  best HR_valid: {}, best_NDCG_valid: {} \n".format(epoch_index, self.best_hr_valid, self.best_ndcg_valid))
                    log("Early stop at {} :  best HR_test: {}, best_NDCG_test: {} \n".format(epoch_index, self.best_hr_test, self.best_ndcg_test))
                    torch.save(best_state_dict, "./saved_model/"+self.config['exp_name']+"_model_parameter_final.pt")
                    early = True
                
                    break
            
                hit_ratio_test, ndcg_test = self.test(init_adjs)
                if hit_ratio_test[0] > self.best_hr_test[0]:
                    best_hit_ratio_test_10 = epoch_index
                if ndcg_test[0] > self.best_ndcg_test[0]:
                    best_ndcg_test_10 = epoch_index
    
                self.best_hr_test = np.maximum(self.best_hr_test, hit_ratio_test)
                self.best_ndcg_test = np.maximum(self.best_ndcg_test, ndcg_test)
                log("best HR_test: {}, best_NDCG_test: {}, best_hit_test_10: {}, best_ndcg_test_10: {} \n".format(self.best_hr_test, self.best_ndcg_test, best_hit_ratio_test_10, best_ndcg_test_10))
                if epoch_index >= 49:
                    log("hit_ratio_valid_50: {}, ndcg_valid_50: {}, hit_ratio_test_50: {}, ndcg_test_50: {} \n".format(hit_ratio_valid_50, ndcg_valid_50, hit_ratio_test_50, ndcg_test_50))    
                    log("best_hit_ratio_valid_50: {}, best_ndcg_valid_50: {}, best_hit_ratio_test_50: {}, best_ndcg_test_50: {} \n".format(best_hit_ratio_valid_50, best_ndcg_valid_50, best_hit_ratio_test_50, best_ndcg_test_50))    
                    log("best_hit_ratio_valid_50_epoch: {}, best_ndcg_valid_50_epoch: {}, best_hit_ratio_test_50_epoch: {}, best_ndcg_test_50_epoch: {} \n".format(best_hit_ratio_valid_50_epoch, best_ndcg_valid_50_epoch, best_hit_ratio_test_50_epoch, best_ndcg_test_50_epoch))    
                    
            if epoch_index == 49:
                hit_ratio_valid_50 = hit_ratio_valid.copy()
                ndcg_valid_50 = ndcg_valid.copy()
                hit_ratio_test_50 = hit_ratio_test.copy()
                ndcg_test_50 = ndcg_test.copy()

                best_hit_ratio_valid_50 = self.best_hr_valid.copy()
                best_ndcg_valid_50 = self.best_ndcg_valid.copy()
                best_hit_ratio_test_50 = self.best_hr_test.copy()
                best_ndcg_test_50 = self.best_ndcg_test.copy()

                best_hit_ratio_valid_50_epoch = best_hit_ratio_valid_10
                best_ndcg_valid_50_epoch = best_ndcg_valid_10
                best_hit_ratio_test_50_epoch = best_hit_ratio_test_10
                best_ndcg_test_50_epoch = best_ndcg_test_10

            
            print("-------------------------------------------\n\n\n")

            self.scheduler.step()
        log("Train finished!")
        if not(early):
            torch.save(best_state_dict, "./saved_model/"+self.config['exp_name']+"_model_parameter_final.pt")

        


    def valid(self, adjs):
        """
        valid model
        """
        log("Validing.....")
        self.model.eval()
    
        is_testing = False

        with torch.no_grad():
            test_kinds_of_embeddings = self.model.encoder(adjs)
        test_user_embedding = test_kinds_of_embeddings['user_embedding'].detach().to(self.device)
        test_item_embedding = test_kinds_of_embeddings['item_embedding'].detach().to(self.device)
    
        hit_ratio = np.zeros(len(self.config['topk']))
        ndcg = np.zeros(len(self.config['topk']))

        if not os.path.exists(self.config['data_dir'] + "sample_valid.txt"):
            all_test_cnt = 0
            user_cnt = 0
            
            store_id_mat_list = []

            for test_user, test_user_item in tqdm(self.valid_loader): 
                all_test_cnt += torch.sum(test_user_item).item()
                user_cnt += len(test_user)
                user_topk, item_compute = self.cal_user_topk(test_user, test_user_embedding, test_item_embedding, \
                    torch.tensor(self.config['coarse_grained_type']), self.config['dim_each_feedback'], self.config['dot_weights'], \
                        self.config['score_choice'], self.config['target_behavior_id'], self.config['all_rank_flag'], is_testing, test_user_item)
                
                cur_hit_cnt = np.array(self.cal_HitRatio_metric(user_topk, test_user_item))
                cur_ndcg = np.array(self.cal_NDCG_metric(user_topk, test_user_item, self.pre_cal_IDCG))
                hit_ratio += cur_hit_cnt
                ndcg += cur_ndcg
                
                cur_store_id_mat = torch.hstack((test_user.reshape((-1,1)), item_compute))
                store_id_mat_list.append(cur_store_id_mat)

 
            store_id_mat = torch.cat(store_id_mat_list, dim = 0).to(torch.int).numpy()
            store_id_mat = store_id_mat[store_id_mat[:,0].argsort()]
            np.savetxt(self.config['data_dir'] + "sample_valid.txt", store_id_mat.astype(int),  fmt='%i', delimiter=',')               
            self.valid_loader.dataset.sample_test_mat = store_id_mat
            hit_ratio = hit_ratio / all_test_cnt
            ndcg = ndcg / user_cnt

            msg = "Valid >> hit_ratio: {}, ndcg: {}".format(hit_ratio, ndcg)
            log(msg)

        else:
            
            num_valid_users = self.valid_loader.dataset.sample_test_mat.shape[0]
            fold_len = num_valid_users // self.config['valid_folds']
            for i_fold in range(self.config['valid_folds']):
                start = i_fold * fold_len
                if i_fold == self.config['valid_folds']-1:
                    end = num_valid_users
                else:
                    end = (i_fold+1)*fold_len
                sub_sample_valid_mat = self.valid_loader.dataset.sample_test_mat[start:end]
                user_topk = self.cal_topk_sample(sub_sample_valid_mat, test_user_embedding, test_item_embedding)
                sub_hit_cnt = self.cal_hit_ratio_sample(user_topk)
                sub_ndcg = self.cal_ndcg_sample(user_topk)
                hit_ratio += sub_hit_cnt
                ndcg += sub_ndcg
            
            hit_ratio = hit_ratio / num_valid_users
            ndcg = ndcg / num_valid_users

            msg = "Valid >> hit_ratio: {}, ndcg: {}".format(hit_ratio, ndcg)
            log(msg)


        return hit_ratio, ndcg
    

    def test(self, adjs):
        """
        test model
        """
        log("Testing.....")
        self.model.eval()
        # self.MLP.eval()
        is_testing = True

        with torch.no_grad():
            test_kinds_of_embeddings = self.model.encoder(adjs)
        test_user_embedding = test_kinds_of_embeddings['user_embedding'].detach().to(self.device)
        test_item_embedding = test_kinds_of_embeddings['item_embedding'].detach().to(self.device)
    
        hit_ratio = np.zeros(len(self.config['topk']))
        ndcg = np.zeros(len(self.config['topk']))

        if not os.path.exists(self.config['data_dir'] + "sample_test.txt"):
            all_test_cnt = 0
            user_cnt = 0
            
            store_id_mat_list = []

            for test_user, test_user_item in tqdm(self.test_loader): 
                all_test_cnt += torch.sum(test_user_item).item()
                user_cnt += len(test_user)
                user_topk, item_compute = self.cal_user_topk(test_user, test_user_embedding, test_item_embedding, \
                    torch.tensor(self.config['coarse_grained_type']), self.config['dim_each_feedback'], self.config['dot_weights'], \
                        self.config['score_choice'], self.config['target_behavior_id'], self.config['all_rank_flag'], is_testing, test_user_item)
                
                cur_hit_cnt = np.array(self.cal_HitRatio_metric(user_topk, test_user_item))
                cur_ndcg = np.array(self.cal_NDCG_metric(user_topk, test_user_item, self.pre_cal_IDCG))
                hit_ratio += cur_hit_cnt
                ndcg += cur_ndcg
                
                cur_store_id_mat = torch.hstack((test_user.reshape((-1,1)), item_compute))
                store_id_mat_list.append(cur_store_id_mat)
    
            store_id_mat = torch.cat(store_id_mat_list, dim = 0).to(torch.int).numpy()
            store_id_mat = store_id_mat[store_id_mat[:,0].argsort()]

            np.savetxt(self.config['data_dir'] + "sample_test.txt", store_id_mat.astype(int),  fmt='%i', delimiter=',')             
            self.test_loader.dataset.sample_test_mat = store_id_mat
            hit_ratio = hit_ratio / all_test_cnt
            ndcg = ndcg / user_cnt

            msg = "Test >> hit_ratio: {}, ndcg: {}".format(hit_ratio, ndcg)
            log(msg)

        else:
         
            num_test_users = self.test_loader.dataset.sample_test_mat.shape[0]
            fold_len = num_test_users // self.config['test_folds']
            for i_fold in range(self.config['test_folds']):
                start = i_fold * fold_len
                if i_fold == self.config['test_folds']-1:
                    end = num_test_users
                else:
                    end = (i_fold+1)*fold_len
                sub_sample_test_mat = self.test_loader.dataset.sample_test_mat[start:end]
                user_topk = self.cal_topk_sample(sub_sample_test_mat, test_user_embedding, test_item_embedding)
                sub_hit_cnt = self.cal_hit_ratio_sample(user_topk)
                sub_ndcg = self.cal_ndcg_sample(user_topk)
                hit_ratio += sub_hit_cnt
                ndcg += sub_ndcg
            
            hit_ratio = hit_ratio / num_test_users
            ndcg = ndcg / num_test_users

            msg = "Test >> hit_ratio: {}, ndcg: {}".format(hit_ratio, ndcg)
            log(msg)


        return hit_ratio, ndcg
    

    def cal_topk_sample(self, sub_sample_mat, user_embedding, item_embedding):
        compute_user = torch.unsqueeze(torch.from_numpy(sub_sample_mat[:,0]), dim = -1)
        compute_user = torch.squeeze(compute_user.repeat(1, 100).reshape((-1,1)))
        compute_item = torch.squeeze(torch.from_numpy(sub_sample_mat[:,1:]).reshape((-1,1)))
        user_embedding = user_embedding.cpu()
        item_embedding = item_embedding.cpu()
        cal_user_embd = user_embedding[compute_user.to(torch.long)].cpu()
        cal_item_embd = item_embedding[compute_item.to(torch.long)].cpu()

        score_positive = torch.zeros(cal_user_embd.shape[0])
        score_negative = torch.zeros(cal_user_embd.shape[0])
        temp_score = torch.zeros(cal_user_embd.shape[0], 2)

        dot_weights = self.config['dot_weights']
        dim_each_feedback = self.config['dim_each_feedback']

        # ========================================================================================================================
        learned_emb_dim = cal_user_embd.shape[1]
        for index in range(int(learned_emb_dim/dim_each_feedback)):
            if index == 0: # pos
                score_positive += torch.sum(torch.mul(cal_user_embd[:, index*dim_each_feedback : (index + 1)*dim_each_feedback], \
                    cal_item_embd[:, index*dim_each_feedback : (index + 1)*dim_each_feedback]), dim = -1)
                
            elif index == 1:  # neg
                score_negative += torch.sum(torch.mul(cal_user_embd[:, index*dim_each_feedback : (index + 1)*dim_each_feedback], \
                    cal_item_embd[:, index*dim_each_feedback : (index + 1)*dim_each_feedback]), dim = -1)


        # ========================================================================================================================


        score_matrix = score_positive - score_negative
            
        score_matrix = score_matrix.reshape((sub_sample_mat.shape[0],-1))
        _, user_topk = torch.topk(score_matrix, self.config['topk'][-1], dim = -1) 

        return user_topk



    def cal_hit_ratio_sample(self, user_topk):
        num_users = user_topk.shape[0]
        real_label = torch.zeros((num_users, 100))
        real_label[:,-1] = 1
        
        hit_cnt_list = []
        for topk in self.config['topk']:
            topk_matrix = torch.zeros((num_users, 100))
            row_index = torch.unsqueeze(torch.arange(0,num_users), dim=-1)
            row_index = row_index.repeat(1,topk).reshape((-1,1))
            col_index = user_topk[:,:topk].reshape((-1,1))
            topk_matrix[row_index[:,0].to(torch.long), col_index[:,0].to(torch.long)] = 1

            hit_cnt = torch.sum(topk_matrix * real_label).cpu().numpy()
            hit_cnt_list.append(hit_cnt)

        return hit_cnt_list



    
    def cal_ndcg_sample(self, user_topk):
        num_users = user_topk.shape[0]
        real_label = torch.zeros((num_users, 100))
        real_label[:,-1] = 1

        NDCG_list = []
        for topk in self.config['topk']:
            # calculate DCG        
            row_index = torch.unsqueeze(torch.arange(0,user_topk.shape[0]), dim=-1)
            row_index = row_index.repeat(1,topk).reshape((-1,1))
            col_index = user_topk[:, :topk].reshape((-1,1))
            topk_rel_matrix = real_label[row_index[:,0].to(torch.long), col_index[:,0].to(torch.long)].reshape((user_topk.shape[0], topk))     # [batch_size, topk], 0/1  0: false 1:true
            pos = torch.arange(1, topk+1)
            DCG = torch.sum(topk_rel_matrix / torch.log2((torch.unsqueeze(pos, dim = 0) + 1)), dim = -1)

            # calculate IDCG
            positive_cnt = torch.sum(real_label, dim = -1)      # [batch_size]
            positive_cnt[torch.squeeze((positive_cnt>=topk).nonzero(), dim = -1)] = topk
            IDCG = self.pre_cal_IDCG[positive_cnt.to(torch.long)]+ 1e-8

            # NDCG
            NDCG = torch.sum(DCG.cpu() / IDCG).numpy()
            NDCG_list.append(NDCG)

        return NDCG_list



    
    def cat_user_item_emb(self, user_embedding, item_embedding, user_item_interaction):
        """
        concatenate user and item embeddings from the learned embeddings w.r.t. user_item_interaction
        params:
            user_item_interaction: [num_interactions, 2]   first colomn is user id, second colomn is item id
        
        return:
            concatenated embedding which will be fed into MLP    [2*emb_dim, num_interactions] 
        """
        selected_user_embedding = user_embedding[user_item_interaction[:,0]]
        selected_item_embedding = item_embedding[user_item_interaction[:,1]]
        cat_embedding = torch.cat((selected_user_embedding, selected_item_embedding), dim = 1)

        return cat_embedding


    def inner_product(self, a, b):
        return torch.sum(a*b, dim=-1)


    def cal_log_loss(self, pred_scores, multi_beh_label, multi_beh_id, beh_weights_map):
        """
        calculate log loss
        params:
            pred_scores: [batch_size] tensor, Calculated from MLP
            multi_beh_label: [batch_size].  label of each user - item interaction. 0 / 1  neg / pos
            multi_beh_id: [batch_size]. beh_id of each user - item interaction.  
            beh_weights_map: [num_behs]. each element is the weight of a specific beh  
        return:
            log loss
        """
        temp_loss = torch.log(pred_scores * multi_beh_label + (1 - pred_scores) * (1-multi_beh_label))
        loss_weights = beh_weights_map[multi_beh_id]
        log_loss = -torch.sum(loss_weights * temp_loss) / torch.sum(loss_weights)
        

        return log_loss



    def cal_BPR_loss(self, users_id, pos_items_id, neg_items_id, user_embeddings_each_behavior, item_embeddings_each_behavior):
        """
        use batch data to calculate BPR loss 
        params:
            users_id: users id in current batch   (batch_size, num_behaviors)
            pos_items_id: positive items w.r.t users    (batch_size, num_behaviors)
            neg_items_id: negative items w.r.t users    (batch_size, num_behaviors)
 
        return:
            BPR_loss_each_beh_matrix: [num_behaviors]  BPR_loss for each behavior
        """    
        num_behaviors = user_embeddings_each_behavior.shape[0]

        batch_user_each_behavior = []
        batch_pos_item_each_behavior = []
        batch_neg_item_each_behavior = []
        for i in range(num_behaviors):
            batch_user_each_behavior.append(user_embeddings_each_behavior[i][users_id[:,i].to(torch.long)])            # [batch_size, emb_dim]
            batch_pos_item_each_behavior.append(item_embeddings_each_behavior[i][pos_items_id[:,i].to(torch.long)])    # [batch_size, emb_dim]
            batch_neg_item_each_behavior.append(item_embeddings_each_behavior[i][neg_items_id[:,i].to(torch.long)])  # [batch_size, emb_dim]

        batch_user_each_behavior = torch.stack(batch_user_each_behavior, dim = 0)           # [num_behaviors, batch_size, emb_dim]       
        batch_pos_item_each_behavior = torch.stack(batch_pos_item_each_behavior, dim = 0)   # [num_behaviors, batch_size, emb_dim]
        batch_neg_item_each_behavior = torch.stack(batch_neg_item_each_behavior, dim = 0)   # [num_behaviors, batch_size, emb_dim]

        pos_ratings = self.inner_product(batch_user_each_behavior, batch_pos_item_each_behavior)   # [num_behaviors, batch_size]   
        neg_ratings = self.inner_product(batch_user_each_behavior, batch_neg_item_each_behavior)   # [num_behaviors, batch_size]
        
        BPR_loss_each_beh_matrix = torch.sum(- F.logsigmoid(pos_ratings - neg_ratings), dim = -1)  # [num_behaviors]

        BPR_loss = torch.sum(BPR_loss_each_beh_matrix)

        return BPR_loss     
    
    

    def cal_BPR_loss_target(self, users_id_target, pos_items_id_target, neg_items_id_target, user_embeddings, item_embeddings):
        """
        use batch data to calculate BPR loss 
        params:
            users_id: users id in current batch   (batch_size, num_behaviors)
            pos_items_id: positive items w.r.t users    (batch_size, num_behaviors)
            neg_items_id: negative items w.r.t users    (batch_size, num_behaviors)
 
        return:
            BPR_loss for target behavior
        """    
        dot_weights = self.config['dot_weights']
        dim_each_feedback = self.config['dim_each_feedback']
        pos_ratings = torch.zeros(len(users_id_target)).to(self.device)
        neg_ratings = torch.zeros(len(users_id_target)).to(self.device)
        cal_user_embd = user_embeddings[users_id_target.to(torch.long)]
        cal_item_embd_pos = item_embeddings[pos_items_id_target.to(torch.long)]
        cal_item_embd_neg = item_embeddings[neg_items_id_target.to(torch.long)]
    
        
        # for index, type in enumerate(self.config['coarse_grained_type']):
        #     if type <= 1:  # positive feedback
        #         pos_ratings += dot_weights[index]*torch.sum(torch.mul(cal_user_embd[:, index*dim_each_feedback : (index + 1)*dim_each_feedback], \
        #             cal_item_embd_pos[:, index*dim_each_feedback : (index + 1)*dim_each_feedback]), dim = -1)

        #         neg_ratings += dot_weights[index]*torch.sum(torch.mul(cal_user_embd[:, index*dim_each_feedback : (index + 1)*dim_each_feedback], \
        #             cal_item_embd_neg[:, index*dim_each_feedback : (index + 1)*dim_each_feedback]), dim = -1)

        #     else: 
        #         pos_ratings -= dot_weights[index]*torch.sum(torch.mul(cal_user_embd[:, index*dim_each_feedback : (index + 1)*dim_each_feedback], \
        #             cal_item_embd_pos[:, index*dim_each_feedback : (index + 1)*dim_each_feedback]), dim = -1)

        #         neg_ratings -= dot_weights[index]*torch.sum(torch.mul(cal_user_embd[:, index*dim_each_feedback : (index + 1)*dim_each_feedback], \
        #             cal_item_embd_neg[:, index*dim_each_feedback : (index + 1)*dim_each_feedback]), dim = -1)

        # ========================================================================================================================
        learned_emb_dim = user_embeddings.shape[1]
        for index in range(int(learned_emb_dim/dim_each_feedback)):
            if index == 0: # pos
                pos_ratings += torch.sum(torch.mul(cal_user_embd[:, index*dim_each_feedback : (index + 1)*dim_each_feedback], \
                    cal_item_embd_pos[:, index*dim_each_feedback : (index + 1)*dim_each_feedback]), dim = -1)
                
                neg_ratings += torch.sum(torch.mul(cal_user_embd[:, index*dim_each_feedback : (index + 1)*dim_each_feedback], \
                    cal_item_embd_neg[:, index*dim_each_feedback : (index + 1)*dim_each_feedback]), dim = -1)

            elif index == 1:  # neg
                pos_ratings -= torch.sum(torch.mul(cal_user_embd[:, index*dim_each_feedback : (index + 1)*dim_each_feedback], \
                    cal_item_embd_pos[:, index*dim_each_feedback : (index + 1)*dim_each_feedback]), dim = -1)

                neg_ratings -= torch.sum(torch.mul(cal_user_embd[:, index*dim_each_feedback : (index + 1)*dim_each_feedback], \
                    cal_item_embd_neg[:, index*dim_each_feedback : (index + 1)*dim_each_feedback]), dim = -1)
        # ========================================================================================================================

        BPR_loss = torch.sum(- F.logsigmoid(pos_ratings - neg_ratings)) 
       # BPR_loss = torch.sum(- F.softplus(pos_ratings-neg_ratings))
        return BPR_loss   


    def cal_BPR_loss_target_aux(self, users_id, pos_items_id, neg_items_id, user_embeddings, item_embeddings, user_embeddings_each_behavior, item_embeddings_each_behavior):
        BPR_loss = 0
        div_cnt = 0  

        for i in range(user_embeddings_each_behavior.shape[0]):
            cur_beh_name = self.config['behavior_name'][i]
            orig_index = self.all_beh_names[self.config['dataset_name']].index(cur_beh_name)
            #print("orig_index=",orig_index)
            orig_pos_item = pos_items_id[:,orig_index]
            used_index = torch.squeeze((orig_pos_item != -1).nonzero(), dim = -1)
            users_id_target = users_id[used_index,orig_index]
            pos_items_id_target = pos_items_id[used_index,orig_index]
            neg_items_id_target = neg_items_id[used_index,orig_index]
            #dot_weights = self.config['dot_weights']
            dim_each_feedback = self.config['dim_each_feedback']
            pos_ratings = torch.zeros(len(users_id_target)).to(self.device)
            neg_ratings = torch.zeros(len(users_id_target)).to(self.device)
        
        
            cal_user_embd = user_embeddings[users_id_target.to(torch.long)]
            cal_item_embd_pos = item_embeddings[pos_items_id_target.to(torch.long)]
            cal_item_embd_neg = item_embeddings[neg_items_id_target.to(torch.long)]
    
            # ========================================================================================================================
            learned_emb_dim = user_embeddings.shape[1]
            for index in range(int(learned_emb_dim/dim_each_feedback)):
                if index == 0: # pos
                    pos_ratings += torch.sum(torch.mul(cal_user_embd[:, index*dim_each_feedback : (index + 1)*dim_each_feedback], \
                        cal_item_embd_pos[:, index*dim_each_feedback : (index + 1)*dim_each_feedback]), dim = -1)
                    
                    neg_ratings += torch.sum(torch.mul(cal_user_embd[:, index*dim_each_feedback : (index + 1)*dim_each_feedback], \
                        cal_item_embd_neg[:, index*dim_each_feedback : (index + 1)*dim_each_feedback]), dim = -1)

                elif index == 1:  # neg
                    pos_ratings -= torch.sum(torch.mul(cal_user_embd[:, index*dim_each_feedback : (index + 1)*dim_each_feedback], \
                        cal_item_embd_pos[:, index*dim_each_feedback : (index + 1)*dim_each_feedback]), dim = -1)

                    neg_ratings -= torch.sum(torch.mul(cal_user_embd[:, index*dim_each_feedback : (index + 1)*dim_each_feedback], \
                        cal_item_embd_neg[:, index*dim_each_feedback : (index + 1)*dim_each_feedback]), dim = -1)

     
            BPR_loss += (self.config['bpr_weight'][i] * torch.sum(- F.logsigmoid(pos_ratings - neg_ratings)))
            
            if self.config['bpr_weight'][i] > 0:
                div_cnt += 1

        BPR_loss = BPR_loss / div_cnt

        return BPR_loss



    def square_euclidean_dist(self, x, y):
        """
        calculate squared norm 2
        Args:
            x: pytorch Variable, with shape [m, d]
            y: pytorch Variable, with shape [n, d]
        Returns:
            dist: pytorch Variable, with shape [m, n]
        """
        m, n = x.size(0), y.size(0)
        
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
      
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        ret = xx + yy
        
        ret.addmm_(1, -2, x, y.t())
        
        ret = ret.clamp(min=1e-20)      # for numerical stability

        return ret



    def cal_Graphlearn_loss(self, out_adjs, user_embeddings, item_embeddings):
        """
        calculate Graph learning loss (graph regularization)
        params: 
            out_adjs: dic type.   u2u_adj, i2i_adj, u2i_adj
            user / tiem _embedding: newly learned user / item embedding 
        """
        if not self.config['sparse_graph_flag']:
        # Graph regularization loss
            graph_loss_u2u = 0
            L_u2u = torch.diagflat(torch.sum(out_adjs['u2u_adj'], -1)) - out_adjs['u2u_adj']
            graph_loss_u2u += self.config['smoothness_ratio'] * torch.trace(torch.mm(user_embeddings.transpose(-1, -2), torch.mm(L_u2u, user_embeddings))) / int(np.prod(out_adjs['u2u_adj'].shape))
            ones_vec_u2u = to_cuda(torch.ones(out_adjs['u2u_adj'].size(-1)), self.device)
            graph_loss_u2u += -self.config['degree_ratio'] * torch.mm(ones_vec_u2u.unsqueeze(0), torch.log(torch.mm(out_adjs['u2u_adj'], ones_vec_u2u.unsqueeze(-1)) + Constants.VERY_SMALL_NUMBER)).squeeze() / out_adjs['u2u_adj'].shape[-1]
            graph_loss_u2u += self.config['sparsity_ratio'] * torch.sum(torch.pow(out_adjs['u2u_adj'], 2)) / int(np.prod(out_adjs['u2u_adj'].shape))


            graph_loss_i2i = 0
            L_i2i = torch.diagflat(torch.sum(out_adjs['i2i_adj'], -1)) - out_adjs['i2i_adj']
            graph_loss_i2i += self.config['smoothness_ratio'] * torch.trace(torch.mm(item_embeddings.transpose(-1, -2), torch.mm(L_i2i, item_embeddings))) / int(np.prod(out_adjs['i2i_adj'].shape))
            ones_vec_i2i = to_cuda(torch.ones(out_adjs['i2i_adj'].size(-1)), self.device)
            graph_loss_i2i += -self.config['degree_ratio'] * torch.mm(ones_vec_i2i.unsqueeze(0), torch.log(torch.mm(out_adjs['i2i_adj'], ones_vec_i2i.unsqueeze(-1)) + Constants.VERY_SMALL_NUMBER)).squeeze() / out_adjs['i2i_adj'].shape[-1]
            graph_loss_i2i += self.config['sparsity_ratio'] * torch.sum(torch.pow(out_adjs['i2i_adj'], 2)) / int(np.prod(out_adjs['i2i_adj'].shape))

            graph_loss = graph_loss_u2u + graph_loss_i2i

        else:
            graph_loss_u2u = self.cal_sparse_graph_loss(user_embeddings, out_adjs['u2u_adj'])
            graph_loss_i2i = self.cal_sparse_graph_loss(item_embeddings, out_adjs['i2i_adj'])

            graph_loss = graph_loss_u2u + graph_loss_i2i


        return graph_loss



    def cal_sparse_graph_loss(self, embeddings, adj):
        graph_loss = 0
        sq_euclidean_dist = self.square_euclidean_dist(embeddings, embeddings)  

        node_index = adj.coalesce().indices()
        sq_euclidean_dist_index = node_index
        sq_euclidean_dist_val = sq_euclidean_dist[node_index[0,:], node_index[1,:]]
        sq_euclidean_dist_sparse = torch.sparse_coo_tensor(sq_euclidean_dist_index, sq_euclidean_dist_val, sq_euclidean_dist.shape)

        graph_loss +=  self.config['smoothness_ratio'] * torch.sparse.sum(adj * sq_euclidean_dist_sparse) / (2 * embeddings.shape[0] * embeddings.shape[0])

        graph_loss += -self.config['degree_ratio'] * torch.sparse.sum(torch.sparse.sum(adj, dim = 1).log1p()) / embeddings.shape[0]

        graph_loss += self.config['sparsity_ratio'] * torch.sparse.sum(adj.pow(2)) / (embeddings.shape[0] * embeddings.shape[0])


        return graph_loss



    def cal_InfoNCE_loss_target(self, user_id, observed_item_id, unobserved_item_id, kinds_of_embeddings):
        """
        use batch data to calculate InfoNCE loss 
        first beh in each feedback is target when conducting fine-grained contrast.   
        IP is target when conducting coarse-grained contrast
        params:
            user_id: sampled used id in this batch
            observed_item_id: sampled observed item id in this batch
            unobserved_item_id: sampled unobserved item id in this batch
            kinds_of_embeddings: user_embeddings_each_behavior, item_embeddings_each_behavior, 
                                 user_embeddings_each_feedback, item_embeddings_each_feedback,
                                 user_embedding, item_embedding
        return: 
            InfoNCE_loss
        """    
        # find unique users and items appear in the batch dataset
        uq_users = torch.unique(user_id)
        uq_items = torch.unique(torch.stack((observed_item_id, unobserved_item_id), dim = 0))
 
        # ---------------------------------- fine-grained contrastive loss ----------------------------------
        fine_grained_InfoNCE_loss_user = self.cal_fine_grained_InfoNCE_loss_target(kinds_of_embeddings['user_embeddings_each_behavior'], uq_users)       # 0 or [4, max_aux_num]
        fine_grained_InfoNCE_loss_item = self.cal_fine_grained_InfoNCE_loss_target(kinds_of_embeddings['item_embeddings_each_behavior'], uq_items)
      
        # ---------------------------------- coarse-grained contrastive loss ----------------------------------
        coarse_grained_InfoNCE_loss_user = self.cal_coarse_grained_InfoNCE_loss_target(kinds_of_embeddings['user_embeddings_each_feedback'], uq_users)   # [3]
        coarse_grained_InfoNCE_loss_item = self.cal_coarse_grained_InfoNCE_loss_target(kinds_of_embeddings['item_embeddings_each_feedback'], uq_items)

        InfoNCE_loss_user = torch.sum(fine_grained_InfoNCE_loss_user) + torch.sum(coarse_grained_InfoNCE_loss_user)
        InfoNCE_loss_item = torch.sum(fine_grained_InfoNCE_loss_item) + torch.sum(coarse_grained_InfoNCE_loss_item)
        
        InfoNCE_loss = InfoNCE_loss_user + InfoNCE_loss_item

        return InfoNCE_loss



    def cal_fine_grained_InfoNCE_loss_target(self, embeddings_each_behavior, uq_index):
        """
        calculate fine-grained InfoNCE loss 
        params:
            embeddings_each_behavior: user / item _embeddings_each_behavior
            uq_index: user / item index in current batch
        return:
            fine-grained InfoNCE loss:  0 or [4, max_aux_num]
        """
        beh_type = torch.tensor(self.config['behavior_type'])    #  (EP/0, IP/1, EN/2, IN/3) of each behavior
        EP_index = torch.squeeze((beh_type == 0).nonzero(), dim = -1)
        IP_index = torch.squeeze((beh_type == 1).nonzero(), dim = -1)
        EN_index = torch.squeeze((beh_type == 2).nonzero(), dim = -1)
        IN_index = torch.squeeze((beh_type == 3).nonzero(), dim = -1)
        
        emb_dim = len(set(beh_type)) * self.config['dim_each_feedback']
        max_inner_num_beh = torch.max(torch.tensor([EP_index.shape[0], IP_index.shape[0], EN_index.shape[0], IN_index.shape[0]]))
        if max_inner_num_beh == 1:
            fine_grained_InfoNCE_loss = 0
        else:
            fine_aux_embeddings = []     # fine-grained auxiliary embeddings
            fine_target_embeddings = []  # fine-grained target embeddings
            mask_padding = []            # mask padding part 
            for i in range(len(torch.tensor(self.config['coarse_grained_type']))):   # EP/0, IP/1, EN/2, IN/3
                cur_fd_aux = embeddings_each_behavior[torch.squeeze((beh_type == torch.tensor(self.config['coarse_grained_type'])[i]).nonzero(), dim = -1)[1:]]          # [aux_num, num_users/items, emb_dim]
                
                # padding with zeros to form unified tensor shape
                orig_size = cur_fd_aux.shape[0]
                cur_fd_aux = self.my_stack(cur_fd_aux, \
                    torch.zeros(max_inner_num_beh - 1 - cur_fd_aux.shape[0], uq_index.shape[0], emb_dim), d = 0)     # [max_aux_num, num_users/items, emb_dim]
                fine_aux_embeddings.append(cur_fd_aux)
                fine_target_embeddings.append(embeddings_each_behavior[torch.squeeze((beh_type == torch.tensor(self.config['coarse_grained_type'])[i]).nonzero(),dim = -1)[0]])

                fd_mask = torch.ones(max_inner_num_beh - 1)
                fd_mask[orig_size:] = 0
                mask_padding.append(fd_mask)
            mask_padding = torch.stack(mask_padding, dim = 0)
            
            fine_aux_embeddings = torch.stack(fine_aux_embeddings, dim = 0)                       # [num_coarse, max_aux_num, num_users/items, emb_dim]
            fine_target_embeddings = torch.stack(fine_target_embeddings, dim = 0)                 # [num_coarse, num_users/items, emb_dim]

            batch_fine_aux_embeddings = fine_aux_embeddings[:,:,uq_index,:]                       # [num_coarse, max_aux_num, batch_num_users/items, emb_dim]
            batch_fine_target_embeddings = fine_target_embeddings[:,:,uq_index,:]                 # [num_coarse, batch_num_users/items, emb_dim]

            if self.config['InfoNCE_loss_batch']:     
                # calculate InfoNCE_loss only between batch nodes
                fine_pos_ratings = self.inner_product(batch_fine_target_embeddings.unsqueeze(1), \
                    batch_fine_aux_embeddings)                                                    # [num_coarse, max_aux_num, batch_num_users/items]

                fine_tot_ratings = torch.matmul(batch_fine_target_embeddings.unsqueeze(1), \
                    torch.transpose(batch_fine_aux_embeddings, -1, -2))                           # [num_coarse, max_aux_num, batch_num_users/items, batch_num_users/items]
        
                fine_ssl_logits = fine_tot_ratings - fine_pos_ratings.unsqueeze(-1)               # [num_coarse, max_aux_num, batch_num_users/items, batch_num_users/items]
                fine_clogits = torch.logsumexp(fine_ssl_logits / self.config['ssl_temp'], dim=-1)           # [num_coarse, max_aux_num, batch_num_users/items]
            else:
                # calculate InfoNCE_loss between batch nodes and all the other nodes
                fine_pos_ratings = self.inner_product(batch_fine_target_embeddings.unsqueeze(1), \
                    batch_fine_aux_embeddings)                                                    # [num_coarse, max_aux_num, batch_num_users/items]

                fine_tot_ratings = torch.matmul(batch_fine_target_embeddings.unsqueeze(1), \
                    torch.transpose(fine_aux_embeddings, -1, -2))                                 # [num_coarse, max_aux_num, batch_num_users/items, num_users/items]
        
                fine_ssl_logits = fine_tot_ratings - fine_pos_ratings.unsqueeze(-1)               # [num_coarse, max_aux_num, batch_num_users/items, num_users/items]
                fine_clogits = torch.logsumexp(fine_ssl_logits / self.config['ssl_temp'], dim=-1)           # [num_coarse, max_aux_num, batch_num_users/items]
            
            fine_grained_InfoNCE_loss = torch.sum(fine_clogits, dim = -1) * mask_padding          # [num_coarse, max_aux_num]  

        return fine_grained_InfoNCE_loss



    def cal_coarse_grained_InfoNCE_loss_target(self, embeddings_each_feedback, uq_index):
        """
        calculate coarse-grained InfoNCE loss 
        let Implicit Positive feedback as target 
        params:
            embeddings_each_behavior: user / item _embeddings_each_feedback
            uq_index: user / item index in current batch
        return:
            coarse-grained InfoNCE loss: [num_other_coarse] tensor
        """
        coarse_grained_type = torch.tensor(self.config['coarse_grained_type'])
        target_coarse_index = None
        other_coarse_index = []
        for id, type in enumerate(coarse_grained_type):
            if type == self.config['target_coarse_id']:
                target_coarse_index = id
            else:
                other_coarse_index.append(id)

        if len(other_coarse_index) == 0:
            return torch.tensor(0)

        coarse_target_embeddings = embeddings_each_feedback[target_coarse_index]    # [num_users/items, emb_dim]
        coarse_aux_embeddings = embeddings_each_feedback[other_coarse_index]        # [num_other_coarse, num_users/items, emb_dim]

        batch_coarse_target_embeddings = coarse_target_embeddings[uq_index]         # [batch_num_users/items, emb_dim]
        batch_coarse_aux_embeddings = coarse_aux_embeddings[:,uq_index,:]           # [num_other_coarse, batch_num_users/items, emb_dim]

        if self.config['InfoNCE_loss_batch']:
            coarse_pos_ratings = self.inner_product(batch_coarse_target_embeddings.unsqueeze(0), \
                    batch_coarse_aux_embeddings)                                                    # [num_other_coarse, batch_num_users/items]
            coarse_tot_ratings = torch.matmul(batch_coarse_target_embeddings.unsqueeze(0), \
                torch.transpose(batch_coarse_aux_embeddings, -1, -2))                               # [num_other_coarse, batch_num_users/items, batch_num_users/items]
    
            coarse_ssl_logits = coarse_tot_ratings - coarse_pos_ratings.unsqueeze(-1)               # [num_other_coarse, batch_num_users/items, batch_num_users/items]
            coarse_clogits = torch.logsumexp(coarse_ssl_logits / self.config['ssl_temp'], dim=-1)             # [num_other_coarse, batch_num_users/items]
        else:
            coarse_pos_ratings = self.inner_product(batch_coarse_target_embeddings.unsqueeze(0), \
                    batch_coarse_aux_embeddings)                                                    # [num_other_coarse, batch_num_users/items]
            coarse_tot_ratings = torch.matmul(batch_coarse_target_embeddings.unsqueeze(0), \
                torch.transpose(coarse_aux_embeddings, -1, -2))                                     # [num_other_coarse, batch_num_users/items, num_users/items]
    
            coarse_ssl_logits = coarse_tot_ratings - coarse_pos_ratings.unsqueeze(-1)               # [num_other_coarse, batch_num_users/items, num_users/items]
            coarse_clogits = torch.logsumexp(coarse_ssl_logits / self.config['ssl_temp'], dim=-1)             # [num_other_coarse, batch_num_users/items]
        
        coarse_grained_InfoNCE_loss = torch.sum(coarse_clogits, dim = -1)                           # [num_other_coarse]

        return coarse_grained_InfoNCE_loss



    def my_stack(self, t1, t2, d):
        if t1.nelement() == 0:
            return t2
        elif t2.nelement() == 0:
            return t1
        else:
            return torch.stack((t1, t2), dim = d)


    
    def cal_InfoNCE_loss_one2one(self, user_id, observed_item_id, unobserved_item_id, kinds_of_embeddings):
        """
        calculate InfoNCE_loss between each pair of behavior or feedback (without target / auxiliary )
        """
        uq_users = torch.unique(user_id)
        uq_items = torch.unique(torch.stack((observed_item_id, unobserved_item_id), dim = 0))
 
        # ---------------------------------- fine-grained contrastive loss ----------------------------------
        fine_grained_InfoNCE_loss_user = self.cal_fine_grained_InfoNCE_loss_one2one(kinds_of_embeddings['user_embeddings_each_behavior'], uq_users)       
        fine_grained_InfoNCE_loss_item = self.cal_fine_grained_InfoNCE_loss_one2one(kinds_of_embeddings['item_embeddings_each_behavior'], uq_items)
      
        # ---------------------------------- coarse-grained contrastive loss ----------------------------------
        coarse_grained_InfoNCE_loss_user = self.cal_coarse_grained_InfoNCE_loss_one2one(kinds_of_embeddings['user_embeddings_each_feedback'], uq_users)   
        coarse_grained_InfoNCE_loss_item = self.cal_coarse_grained_InfoNCE_loss_one2one(kinds_of_embeddings['item_embeddings_each_feedback'], uq_items)
 
        InfoNCE_loss_user = fine_grained_InfoNCE_loss_user + coarse_grained_InfoNCE_loss_user
        InfoNCE_loss_item = fine_grained_InfoNCE_loss_item + coarse_grained_InfoNCE_loss_item

        InfoNCE_loss = InfoNCE_loss_user + InfoNCE_loss_item

        return InfoNCE_loss



    def cal_fine_grained_InfoNCE_loss_one2one(self, embeddings_each_behavior, uq_index):
        fine_grained_InfoNCE_loss = 0

        for i in range(len(torch.tensor(self.config['coarse_grained_type']))):   # EP/0, IP/1, EN/2, IN/3
            cur_fd_size = len(torch.squeeze((torch.tensor(self.config['behavior_type'])== torch.tensor(self.config['coarse_grained_type'])[i]).nonzero(), dim = -1))
            if cur_fd_size == 1:
                continue

            cur_fd = embeddings_each_behavior[torch.squeeze((torch.tensor(self.config['behavior_type']) == torch.tensor(self.config['coarse_grained_type'])[i]).nonzero(), dim = -1)]     # [num_beh_fd, num_users/items, emb_dim]   num_beh_fd: num_beh of this fd
            expand_beh = torch.stack([cur_fd for _ in range(cur_fd_size)], dim = 0)    # [num_beh_fd, num_beh_fd, num_users/items, emb_dim]
            
            batch_emb = cur_fd[:,uq_index,:]                    # [num_beh_fd, batch_num_users/items, emb_dim]
            batch_expand_beh = expand_beh[:,:,uq_index,:]       # [num_beh_fd, num_beh_fd, batch_num_users/items, emb_dim]

            if self.config['InfoNCE_loss_batch']:
                fine_pos_ratings = self.inner_product(batch_emb.unsqueeze(1), batch_expand_beh)                      # [num_beh_fd, num_beh_fd, batch_num_users/items]

                fine_tot_ratings = torch.matmul(batch_emb.unsqueeze(1), torch.transpose(batch_expand_beh, -1, -2))   # [num_beh_fd, num_beh_fd, batch_num_users/items, batch_num_users/items]

                fine_ssl_logits = fine_tot_ratings - fine_pos_ratings.unsqueeze(-1)                                  # [num_beh_fd, num_beh_fd, batch_num_users/items, batch_num_users/items]
                fine_clogits = torch.logsumexp(fine_ssl_logits / self.config['ssl_temp'], dim=-1)                                # [num_beh_fd, num_beh_fd, batch_num_users/items]
        
            else:
                # calculate InfoNCE_loss between batch nodes and all the other nodes
                fine_pos_ratings = self.inner_product(batch_emb.unsqueeze(1), batch_expand_beh)                      # [num_beh_fd, num_beh_fd, batch_num_users/items]

                fine_tot_ratings = torch.matmul(batch_emb.unsqueeze(1), torch.transpose(expand_beh, -1, -2))         # [num_beh_fd, num_beh_fd, batch_num_users/items, num_users/items]

                fine_ssl_logits = fine_tot_ratings - fine_pos_ratings.unsqueeze(-1)                                  # [num_beh_fd, num_beh_fd, batch_num_users/items, num_users/items]
                fine_clogits = torch.logsumexp(fine_ssl_logits / self.config['ssl_temp'], dim=-1)                              # [num_beh_fd, num_beh_fd, batch_num_users/items]
        
            InfoNCE_loss_self = torch.sum(fine_clogits, dim = -1)             # [num_beh_fd, num_beh_fd]
            mask_self = torch.eye(cur_fd_size)
            
            fine_InfoNCE_loss = torch.sum(InfoNCE_loss_self * mask_self)      

            fine_grained_InfoNCE_loss += fine_InfoNCE_loss

        return fine_grained_InfoNCE_loss

    
    def cal_coarse_grained_InfoNCE_loss_one2one(self, embeddings_each_feedback, uq_index): 
        coarse_InfoNCE_loss = 0
        if len(torch.tensor(self.config['coarse_grained_type'])) <= 1:
            return coarse_InfoNCE_loss

        expand_fd = torch.stack([embeddings_each_feedback for _ in range(len(torch.tensor(self.config['coarse_grained_type'])))])    # [num_coarse_type, num_coarse_type, num_users/items, emb_dim]
        batch_fd = embeddings_each_feedback[:,uq_index, :]          # [num_coarse_type, batch_num_users/items, emb_dim]
        batch_expand_fd = expand_fd[:, :, uq_index, :]              # [num_coarse_type, num_coarse_type, batch_num_users/items, emb_dim]


        if self.config['InfoNCE_loss_batch']:
            coarse_pos_ratings = self.inner_product(batch_fd.unsqueeze(1), batch_expand_fd)                       # [num_coarse_type, num_coarse_type, batch_num_users/items]

            coarse_tot_ratings = torch.matmul(batch_fd.unsqueeze(1), torch.transpose(batch_expand_fd, -1, -2))    # [num_coarse_type, num_coarse_type, batch_num_users/items, batch_num_users/items]

            coarse_ssl_logits = coarse_tot_ratings - coarse_pos_ratings.unsqueeze(-1)                             # [num_coarse_type, num_coarse_type, batch_num_users/items, batch_num_users/items]

            coarse_clogits = torch.logsumexp(coarse_ssl_logits / self.config['ssl_temp'], dim=-1)                           # [num_coarse_type, num_coarse_type, batch_num_users/items]
        
        else:
            coarse_pos_ratings = self.inner_product(batch_fd.unsqueeze(1), batch_expand_fd)                       # [num_coarse_type, num_coarse_type, batch_num_users/items]

            coarse_tot_ratings = torch.matmul(batch_fd.unsqueeze(1), torch.transpose(expand_fd, -1, -2))          # [num_coarse_type, num_coarse_type, batch_num_users/items, num_users/items]

            coarse_ssl_logits = coarse_tot_ratings - coarse_pos_ratings.unsqueeze(-1)                             # [num_coarse_type, num_coarse_type, batch_num_users/items, num_users/items]

            coarse_clogits = torch.logsumexp(coarse_ssl_logits / self.config['ssl_temp'], dim=-1)                           # [num_coarse_type, num_coarse_type, batch_num_users/items]

        InfoNCE_loss_self = torch.sum(coarse_clogits, dim = -1)           # [num_coarse_type, num_coarse_type]
        mask_self = torch.eye(4)

        coarse_InfoNCE_loss = torch.sum(InfoNCE_loss_self * mask_self)   

        return coarse_InfoNCE_loss



    def cal_InfoNCE_loss_NN_PP(self, user_id, observed_item_id, unobserved_item_id, kinds_of_embeddings):
        """
        """
        uq_users = torch.unique(user_id)
        uq_items = torch.unique(torch.stack((observed_item_id, unobserved_item_id), dim = 0))
 
        # ---------------------------------- fine-grained contrastive loss ----------------------------------
        fine_grained_InfoNCE_loss_user = self.cal_fine_grained_InfoNCE_loss_one2one(kinds_of_embeddings['user_embeddings_each_behavior'], uq_users)       
        fine_grained_InfoNCE_loss_item = self.cal_fine_grained_InfoNCE_loss_one2one(kinds_of_embeddings['item_embeddings_each_behavior'], uq_items)
      
        # ---------------------------------- coarse-grained contrastive loss ----------------------------------
        coarse_grained_InfoNCE_loss_user = self.cal_coarse_grained_InfoNCE_loss_NN_PP(kinds_of_embeddings['user_embeddings_each_feedback'], uq_users)   
        coarse_grained_InfoNCE_loss_item = self.cal_coarse_grained_InfoNCE_loss_NN_PP(kinds_of_embeddings['item_embeddings_each_feedback'], uq_items)
 
        InfoNCE_loss_user = fine_grained_InfoNCE_loss_user + coarse_grained_InfoNCE_loss_user
        InfoNCE_loss_item = fine_grained_InfoNCE_loss_item + coarse_grained_InfoNCE_loss_item

        InfoNCE_loss = InfoNCE_loss_user + InfoNCE_loss_item

        return InfoNCE_loss



    def cal_coarse_grained_InfoNCE_loss_NN_PP(self, embeddings_each_feedback, uq_index):
        """
        conduct contrastive learning within EP/IP and EN/IN 
        """
        
        coarse_InfoNCE_loss = 0
        if len(torch.tensor(self.config['coarse_grained_type'])) <= 1:
            return coarse_InfoNCE_loss


        if self.config['InfoNCE_loss_batch']:
            if 0 in torch.tensor(self.config['coarse_grained_type']) and 1 in torch.tensor(self.config['coarse_grained_type']): 
                # EP, IP
                EP_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type'])==0).nonzero(), dim = -1)
                EP_embeddings_batch = embeddings_each_feedback[EP_index, uq_index]     # [batch_size, emb_dim]

                IP_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type'])==1).nonzero(), dim = -1)
                IP_embeddings_batch = embeddings_each_feedback[IP_index, uq_index]     # [batch_size, emb_dim]

                coarse_pos_ratings_EP_IP = self.inner_product(EP_embeddings_batch, IP_embeddings_batch)               # [batch_size]
                coarse_tot_ratings_EP_IP = torch.matmul(EP_embeddings_batch, IP_embeddings_batch)                     # [batch_size, batch_size]
                coarse_ssl_logits_EP_IP = coarse_tot_ratings_EP_IP - coarse_pos_ratings_EP_IP.unsqueeze(-1)           # [batch_size, batch_size]
                coarse_clogits_EP_IP = torch.logsumexp(coarse_ssl_logits_EP_IP / self.config['ssl_temp'], dim=-1)     # [batch_size]

                coarse_tot_ratings_IP_EP = torch.transpose(coarse_tot_ratings_EP_IP, -1, -2)
                coarse_ssl_logits_IP_EP = coarse_tot_ratings_IP_EP - coarse_pos_ratings_EP_IP.unsqueeze(-1)           # [batch_size, batch_size]
                coarse_clogits_IP_EP = torch.logsumexp(coarse_ssl_logits_IP_EP / self.config['ssl_temp'], dim=-1)     # [batch_size]
                coarse_InfoNCE_loss += torch.sum(coarse_clogits_IP_EP)

            if 2 in torch.tensor(self.config['coarse_grained_type']) and 3 in torch.tensor(self.config['coarse_grained_type']):
                # EN, IN
                EN_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type'])==2).nonzero(), dim = -1)
                EN_embeddings_batch = embeddings_each_feedback[EN_index, uq_index]     # [batch_size, emb_dim]

                IN_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type'])==3).nonzero(), dim = -1)
                IN_embeddings_batch = embeddings_each_feedback[IN_index, uq_index]     # [batch_size, emb_dim]

                coarse_pos_ratings_EN_IN = self.inner_product(EN_embeddings_batch, IN_embeddings_batch)               # [batch_size]
                coarse_tot_ratings_EN_IN = torch.matmul(EN_embeddings_batch, IN_embeddings_batch)                     # [batch_size, batch_size]
                coarse_ssl_logits_EN_IN = coarse_tot_ratings_EN_IN - coarse_pos_ratings_EN_IN.unsqueeze(-1)           # [batch_size, batch_size]
                coarse_clogits_EN_IN = torch.logsumexp(coarse_ssl_logits_EN_IN / self.config['ssl_temp'], dim=-1)     # [batch_size]
                coarse_InfoNCE_loss += torch.sum(coarse_clogits_EN_IN)

                coarse_tot_ratings_IN_EN = torch.transpose(coarse_tot_ratings_EN_IN, -1, -2)
                coarse_ssl_logits_IN_EN = coarse_tot_ratings_IN_EN - coarse_pos_ratings_EN_IN.unsqueeze(-1)           # [batch_size, batch_size]
                coarse_clogits_IN_EN = torch.logsumexp(coarse_ssl_logits_IN_EN / self.config['ssl_temp'], dim=-1)     # [batch_size]
                coarse_InfoNCE_loss += torch.sum(coarse_clogits_IN_EN)

        else:
            if 0 in torch.tensor(self.config['coarse_grained_type']) and 1 in torch.tensor(self.config['coarse_grained_type']): 
                # EP, IP
                EP_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type'])==0).nonzero(), dim = -1)
                EP_embeddings_batch = embeddings_each_feedback[EP_index, uq_index]     # [batch_size, emb_dim]
                EP_embeddings = embeddings_each_feedback[EP_index]     # [num_users/items, emb_dim]

                IP_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type'])==1).nonzero(), dim = -1)
                IP_embeddings_batch = embeddings_each_feedback[IP_index, uq_index]     # [batch_size, emb_dim]
                IP_embeddings = embeddings_each_feedback[IP_index]     # [num_users/items, emb_dim]

                coarse_pos_ratings_EP_IP = self.inner_product(EP_embeddings_batch, IP_embeddings_batch)               # [batch_size]
                coarse_tot_ratings_EP_IP = torch.matmul(EP_embeddings_batch, IP_embeddings)                           # [batch_size, num_users/items]
                coarse_ssl_logits_EP_IP = coarse_tot_ratings_EP_IP - coarse_pos_ratings_EP_IP.unsqueeze(-1)           # [batch_size, num_users/items]
                coarse_clogits_EP_IP = torch.logsumexp(coarse_ssl_logits_EP_IP / self.config['ssl_temp'], dim=-1)     # [batch_size]
                coarse_InfoNCE_loss += torch.sum(coarse_clogits_EP_IP)
                
                coarse_pos_ratings_IP_EP = self.inner_product(IP_embeddings_batch, EP_embeddings_batch)               # [batch_size]
                coarse_tot_ratings_IP_EP = torch.matmul(IP_embeddings_batch, EP_embeddings)                           # [batch_size, num_users/items]
                coarse_ssl_logits_IP_EP = coarse_tot_ratings_IP_EP - coarse_pos_ratings_IP_EP.unsqueeze(-1)           # [batch_size, num_users/items]
                coarse_clogits_IP_EP = torch.logsumexp(coarse_ssl_logits_IP_EP / self.config['ssl_temp'], dim=-1)     # [batch_size]
                coarse_InfoNCE_loss += torch.sum(coarse_clogits_IP_EP)

        
            if 2 in torch.tensor(self.config['coarse_grained_type']) and 3 in torch.tensor(self.config['coarse_grained_type']):
                # EN, IN
                EN_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type'])==0).nonzero(), dim = -1)
                EN_embeddings_batch = embeddings_each_feedback[EN_index, uq_index]     # [batch_size, emb_dim]
                EN_embeddings = embeddings_each_feedback[EN_index]     # [num_users/items, emb_dim]

                IN_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type'])==1).nonzero(), dim = -1)
                IN_embeddings_batch = embeddings_each_feedback[IN_index, uq_index]     # [batch_size, emb_dim]
                IN_embeddings = embeddings_each_feedback[IN_index]     # [num_users/items, emb_dim]

                coarse_pos_ratings_EN_IN = self.inner_product(EN_embeddings_batch, IN_embeddings_batch)               # [batch_size]
                coarse_tot_ratings_EN_IN = torch.matmul(EN_embeddings_batch, IN_embeddings)                           # [batch_size, num_users/items]
                coarse_ssl_logits_EN_IN = coarse_tot_ratings_EN_IN - coarse_pos_ratings_EN_IN.unsqueeze(-1)           # [batch_size, num_users/items]
                coarse_clogits_EN_IN = torch.logsumexp(coarse_ssl_logits_EN_IN / self.config['ssl_temp'], dim=-1)     # [batch_size]
                coarse_InfoNCE_loss += torch.sum(coarse_clogits_EN_IN)
                
                coarse_pos_ratings_IN_EN = self.inner_product(IN_embeddings_batch, EN_embeddings_batch)               # [batch_size]
                coarse_tot_ratings_IN_EN = torch.matmul(IN_embeddings_batch, EN_embeddings)                           # [batch_size, num_users/items]
                coarse_ssl_logits_IN_EN = coarse_tot_ratings_IN_EN - coarse_pos_ratings_IN_EN.unsqueeze(-1)           # [batch_size, num_users/items]
                coarse_clogits_IN_EN = torch.logsumexp(coarse_ssl_logits_IN_EN / self.config['ssl_temp'], dim=-1)     # [batch_size]
                coarse_InfoNCE_loss += torch.sum(coarse_clogits_IN_EN)

        return coarse_InfoNCE_loss   


    def cal_InfoNCE_loss_paper(self, user_id, observed_item_id, unobserved_item_id, kinds_of_embeddings):
        """
        calculate InfoNCE_loss as paper said
        """

        uq_users = user_id
        uq_items = observed_item_id    
     
        #print("cur_batch user,item",len(uq_users),len(uq_items))
        # ---------------------------------- fine-grained contrastive loss ----------------------------------
        if self.config['fine_contrast']:
            if self.config['wo_sample']:
                fine_grained_InfoNCE_loss_user = self.cal_fine_grained_InfoNCE_loss_paper_wo_sample_for(kinds_of_embeddings['user_embeddings_each_behavior'], uq_users, 'user')       
                fine_grained_InfoNCE_loss_item = self.cal_fine_grained_InfoNCE_loss_paper_wo_sample_for(kinds_of_embeddings['item_embeddings_each_behavior'], uq_items, 'item')
            else:
                fine_grained_InfoNCE_loss_user = self.cal_fine_grained_InfoNCE_loss_paper(kinds_of_embeddings['user_embeddings_each_behavior'], uq_users)       
                fine_grained_InfoNCE_loss_item = self.cal_fine_grained_InfoNCE_loss_paper(kinds_of_embeddings['item_embeddings_each_behavior'], uq_items)
        else:
            fine_grained_InfoNCE_loss_user = torch.tensor(0)     
            fine_grained_InfoNCE_loss_item = torch.tensor(0)
            
        if self.config['coarse_contrast']:
            # ---------------------------------- coarse-grained contrastive loss ----------------------------------
            if self.config['wo_sample']:
                coarse_grained_InfoNCE_loss_user = self.cal_coarse_grained_InfoNCE_loss_paper_wo_sample(kinds_of_embeddings['user_embeddings_each_feedback'], uq_users, 'user')   
                coarse_grained_InfoNCE_loss_item = self.cal_coarse_grained_InfoNCE_loss_paper_wo_sample(kinds_of_embeddings['item_embeddings_each_feedback'], uq_items, 'item')
            else:
                coarse_grained_InfoNCE_loss_user = self.cal_coarse_grained_InfoNCE_loss_paper(kinds_of_embeddings['user_embeddings_each_feedback'], uq_users)   
                coarse_grained_InfoNCE_loss_item = self.cal_coarse_grained_InfoNCE_loss_paper(kinds_of_embeddings['item_embeddings_each_feedback'], uq_items)
        else:
            coarse_grained_InfoNCE_loss_user = torch.tensor(0)
            coarse_grained_InfoNCE_loss_item = torch.tensor(0)
            #print("meiyou coarse!")
      

        InfoNCE_loss_user = fine_grained_InfoNCE_loss_user + self.config['coarse_infonce_coe'] * coarse_grained_InfoNCE_loss_user
        InfoNCE_loss_item = fine_grained_InfoNCE_loss_item + self.config['coarse_infonce_coe'] * coarse_grained_InfoNCE_loss_item
        
        #print("fine_infonce_loss:", fine_grained_InfoNCE_loss_user,fine_grained_InfoNCE_loss_item)
        #print("coarse_infonce_loss:", coarse_grained_InfoNCE_loss_user, coarse_grained_InfoNCE_loss_item)        

        InfoNCE_loss = InfoNCE_loss_user + InfoNCE_loss_item
        # InfoNCE_loss = InfoNCE_loss_user
    
      
        return InfoNCE_loss


    def cal_fine_grained_InfoNCE_loss_paper_wo_sample(self, embeddings_each_behavior, uq_index):
        # for循环的形式，没有expand
        fine_grained_InfoNCE_loss = torch.tensor(0).to(torch.float).to(self.device)
        infonce_fold = self.config['infonce_fold']

        for i in range(len(torch.tensor(self.config['coarse_grained_type']))):   # EP/0, IP/1, EN/2, IN/3
            if torch.tensor(self.config['coarse_grained_type'])[i] != self.config['target_coarse_id']:
                ## ------------------------ auxiliary behavior one2one -------------------------
                cur_fd_size = len(torch.squeeze((torch.tensor(self.config['behavior_type'])== torch.tensor(self.config['coarse_grained_type'])[i]).nonzero(), dim = -1))
                if cur_fd_size == 1:
                    continue
                
                cur_fd = embeddings_each_behavior[torch.squeeze((torch.tensor(self.config['behavior_type']) == torch.tensor(self.config['coarse_grained_type'])[i]).nonzero(), dim = -1)]     # [num_beh_fd, num_users/items, emb_dim]   num_beh_fd: num_beh of this fd
                for j in range(len(cur_fd.shape[0])):
                    beh_emb = cur_fd[j][uq_index.to(torch.long)]
                    other_beh_emb = []
                    for k in range(len(cur_fd.shape[0])):
                        if k!=j:
                            other_beh_emb.append(cur_fd[k][uq_index.to(torch.long)])
                    other_beh_emb = torch.stack(other_beh_emb, dim = 0)   # [num_aux, num_batch, emb_dim]
  
                    fold_len = len(uq_index) // infonce_fold
                    for i_fold in range(int(infonce_fold)):
                        start = i_fold * fold_len
                        if i_fold == infonce_fold - 1:
                            end = len(uq_index)
                        else:
                            end = (i_fold+1)*fold_len
  
                        cur_beh_emb = beh_emb[start:end]      # [fold_len, emb_dim]
                        fine_pos_ratings = self.inner_product(cur_beh_emb.unsqueeze(0), other_beh_emb[:,start:end,:])      # [num_other, fold_len]              

                        fine_tot_ratings = torch.matmul(cur_beh_emb.unsqueeze(0), torch.transpose(other_beh_emb, -1, -2))    # [num_other, fold_len, num_batch]        

                        fine_ssl_logits = fine_tot_ratings - fine_pos_ratings.unsqueeze(-1)  
                        fine_clogits = torch.logsumexp(fine_ssl_logits / self.config['ssl_temp'], dim=-1)                                   
                        fine_grained_InfoNCE_loss += torch.sum(fine_clogits)

            else:
                ## ------------------------ target behavior -------------------------
                cur_fd_size = len(torch.squeeze((torch.tensor(self.config['behavior_type'])== torch.tensor(self.config['coarse_grained_type'])[i]).nonzero(), dim = -1))
                if cur_fd_size == 1:
                    continue
                    
                cur_fd = embeddings_each_behavior[torch.squeeze((torch.tensor(self.config['behavior_type']) == torch.tensor(self.config['coarse_grained_type'])[i]).nonzero(), dim = -1)]     # [num_beh_fd, num_users/items, emb_dim]   num_beh_fd: num_beh of this fd
                target_emb = cur_fd[0][uq_index.to(torch.long)]  # [num_batch, emb_dim]
                aux_emb = cur_fd[1:,uq_index.to(torch.long),:]    # [num_aux, num_batch, emb_dim]

                fold_len = len(uq_index) // infonce_fold
                for i_fold in range(int(infonce_fold)):
                    start = i_fold * fold_len
                    if i_fold == infonce_fold - 1:
                        end = len(uq_index)
                    else:
                        end = (i_fold+1)*fold_len

                    cur_beh_emb = target_emb[start:end]      # [fold_len, emb_dim]
                    fine_pos_ratings = self.inner_product(cur_beh_emb.unsqueeze(0), aux_emb[:,start:end,:])      # [num_aux, fold_len]              

                    fine_tot_ratings = torch.matmul(cur_beh_emb.unsqueeze(0), torch.transpose(aux_emb, -1, -2))    # [num_aux, fold_len, num_batch]        

                    fine_ssl_logits = fine_tot_ratings - fine_pos_ratings.unsqueeze(-1)  
                    fine_clogits = torch.logsumexp(fine_ssl_logits / self.config['ssl_temp'], dim=-1)                                   
                    fine_grained_InfoNCE_loss += torch.sum(fine_clogits)

        return fine_grained_InfoNCE_loss



    def cal_fine_grained_InfoNCE_loss_paper_wo_sample_for(self, embeddings_each_behavior, uq_index, node_type):
    
        fine_grained_InfoNCE_loss = torch.tensor(0).to(torch.float).to(self.device)
        infonce_fold = self.config['infonce_fold']

        uq_index = torch.as_tensor(np.random.choice(uq_index.cpu(), size=int(len(uq_index)/10), replace=False, p=None)).to(self.config['device'])

        for i in range(len(torch.tensor(self.config['coarse_grained_type']))):   # EP/0, IP/1, EN/2, IN/3
            if torch.tensor(self.config['coarse_grained_type'])[i] != self.config['target_coarse_id']:
                ## ------------------------ auxiliary behavior one2one -------------------------
                cur_fd_size = len(torch.squeeze((torch.tensor(self.config['behavior_type'])== torch.tensor(self.config['coarse_grained_type'])[i]).nonzero(), dim = -1))
                if cur_fd_size == 1:
                    continue
                
              
 
                cur_fd_index = torch.squeeze((torch.tensor(self.config['behavior_type']) == torch.tensor(self.config['coarse_grained_type'])[i]).nonzero(), dim = -1)
                cur_fd = embeddings_each_behavior[cur_fd_index]     # [num_beh_fd, num_users/items, emb_dim]   num_beh_fd: num_beh of this fd
                div_cnt = cur_fd.shape[0] * (cur_fd.shape[0] - 1)   

                for j in range(cur_fd.shape[0]):
                    beh_name = self.config['behavior_name'][cur_fd_index[j]]
                    # beh_emb = cur_fd[j][uq_index.to(torch.long)]
                    other_beh_emb = None
                    for k in range(cur_fd.shape[0]):
                        if k != j:    
                            other_beh_name = self.config['behavior_name'][cur_fd_index[k]]
                            
                            # get the interaction node set of behaviors
                            if os.path.exists(self.config['data_dir']+beh_name + '_' + other_beh_name + "_" + node_type + '.txt'):
                                inter_node_set = np.loadtxt(self.config['data_dir']+beh_name + '_' + other_beh_name + "_" + node_type + '.txt', dtype=int)
                            else:
                                inter_node_set = np.loadtxt(self.config['data_dir']+ other_beh_name + '_' + beh_name + "_" + node_type + '.txt', dtype=int)

                            inter_cnt = len(inter_node_set)

                            if inter_cnt == 0:
                                continue
                            
                            np_uq_index = uq_index.cpu().numpy()
                            tri_inter = np.intersect1d(np_uq_index, inter_node_set,  assume_unique=True)
                            tri_cnt = len(tri_inter)
                            
                            if tri_cnt == 0:
                                continue
                            tri_inter = torch.from_numpy(tri_inter)
                        
                            beh_emb = cur_fd[j][tri_inter.to(torch.long)]
                            other_beh_emb = cur_fd[k][tri_inter.to(torch.long)]

                            # other_beh_emb = cur_fd[k][uq_index.to(torch.long)]
                            fold_len = len(tri_inter) // infonce_fold
                            for i_fold in range(int(infonce_fold)):
                                start = i_fold * fold_len
                                if i_fold == infonce_fold - 1:
                                    end = len(tri_inter)
                                else:
                                    end = (i_fold+1)*fold_len
                                cur_beh_emb = beh_emb[start:end]      # [fold_len, emb_dim]
                                fine_pos_ratings = self.inner_product(cur_beh_emb, other_beh_emb[start:end,:])               
                                fine_tot_ratings = torch.matmul(cur_beh_emb, torch.transpose(other_beh_emb, -1, -2))    # [num_other, fold_len, num_batch]        

                                fine_ssl_logits = fine_tot_ratings - fine_pos_ratings.unsqueeze(-1)  
                                fine_clogits = torch.logsumexp(fine_ssl_logits / self.config['ssl_temp'], dim=-1)                                   
                                fine_grained_InfoNCE_loss += (self.config['aux_infonce_coe']*torch.sum(fine_clogits) / div_cnt)

                            

                ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            else:
                ## ------------------------ target behavior -------------------------
                cur_fd_size = len(torch.squeeze((torch.tensor(self.config['behavior_type'])== torch.tensor(self.config['coarse_grained_type'])[i]).nonzero(), dim = -1))
                if cur_fd_size == 1:
                    continue
                    
                cur_fd = embeddings_each_behavior[torch.squeeze((torch.tensor(self.config['behavior_type']) == torch.tensor(self.config['coarse_grained_type'])[i]).nonzero(), dim = -1)]     # [num_beh_fd, num_users/items, emb_dim]   num_beh_fd: num_beh of this fd
                target_emb = cur_fd[0][uq_index.to(torch.long)]  # [num_batch, emb_dim]
                aux_emb = cur_fd[0:,uq_index.to(torch.long),:]    # [num_aux, num_batch, emb_dim]
                
                div_cnt = aux_emb.shape[0]

                for k in range(aux_emb.shape[0]):
                    fold_len = len(uq_index) // infonce_fold
                    for i_fold in range(int(infonce_fold)):
                        start = i_fold * fold_len
                        if i_fold == infonce_fold - 1:
                            end = len(uq_index)
                        else:
                            end = (i_fold+1)*fold_len

                        cur_beh_emb = target_emb[start:end]      # [fold_len, emb_dim]
                        fine_pos_ratings = self.inner_product(cur_beh_emb, aux_emb[k,start:end,:])      # [fold_len]              
                        
                        cur_fold_index = set(np.array(uq_index[start:end].cpu()))
                        batch_index = set(np.array(uq_index.cpu()))
                        neg_index_set = batch_index - cur_fold_index
                        neg_index = torch.as_tensor(np.array(list(neg_index_set))).long().to(self.config['device'])
                        
                        fine_tot_ratings = torch.matmul(cur_beh_emb, torch.transpose(cur_fd[k,neg_index], -1, -2))    # [fold_len, num_batch]        

                        fine_ssl_logits = fine_tot_ratings - fine_pos_ratings.unsqueeze(-1)  
                        fine_clogits = torch.logsumexp(fine_ssl_logits / self.config['ssl_temp_list'][k], dim=-1)                                   
                        fine_grained_InfoNCE_loss += (torch.sum(fine_clogits) / div_cnt)

        # fine_grained_InfoNCE_loss = fine_grained_InfoNCE_loss / len(self.config['behavior_name'])
        return fine_grained_InfoNCE_loss





    def cal_fine_grained_InfoNCE_loss_paper_v2(self, embeddings_each_behavior, uq_index):
        # for循环的形式，没有expand
        fine_grained_InfoNCE_loss = torch.tensor(0).to(torch.float).to(self.device)
        
        for i in range(len(torch.tensor(self.config['coarse_grained_type']))):   # EP/0, IP/1, EN/2, IN/3
            if torch.tensor(self.config['coarse_grained_type'])[i] != self.config['target_coarse_id']:
                ## ------------------------ auxiliary behavior one2one -------------------------
                cur_fd_size = len(torch.squeeze((torch.tensor(self.config['behavior_type'])== torch.tensor(self.config['coarse_grained_type'])[i]).nonzero(), dim = -1))
                if cur_fd_size == 1:
                    continue
                
                sample_size = int(len(uq_index)*self.config['sample_rate'])
                sample_index = uq_index[torch.randperm(len(uq_index))[:sample_size].to(torch.long)]

                cur_fd = embeddings_each_behavior[torch.squeeze((torch.tensor(self.config['behavior_type']) == torch.tensor(self.config['coarse_grained_type'])[i]).nonzero(), dim = -1)]     # [num_beh_fd, num_users/items, emb_dim]   num_beh_fd: num_beh of this fd
                for j in range(len(cur_fd.shape[0])):
                    beh_emb = cur_fd[j]
                    other_beh_emb = []
                    for k in range(len(cur_fd.shape[0])):
                        if k!=j:
                            other_beh_emb.append(cur_fd[k])
                    other_beh_emb = torch.stack(other_beh_emb, dim = 0)

                    if self.config['sample_infonce']:
                        sample_beh_emb = beh_emb[sample_index]
                        sample_other_beh_emb = other_beh_emb[:,sample_index]
                        fine_pos_ratings = self.inner_product(sample_beh_emb.unsqueeze(0), sample_other_beh_emb)                      # [num_beh_fd, num_beh_fd, batch_num_users/items]

                        fine_tot_ratings = torch.matmul(sample_beh_emb.unsqueeze(0), torch.transpose(sample_other_beh_emb, -1, -2))   # [num_beh_fd, num_beh_fd, batch_num_users/items, batch_num_users/items]

                        fine_ssl_logits = fine_tot_ratings - fine_pos_ratings.unsqueeze(-1)                                  # [num_beh_fd, num_beh_fd, batch_num_users/items, batch_num_users/items]
                        fine_clogits = torch.logsumexp(fine_ssl_logits / self.config['ssl_temp'], dim=-1)      
                    fine_grained_InfoNCE_loss += torch.sum(fine_clogits)

            else:
                ## ------------------------ target behavior -------------------------
                cur_fd_size = len(torch.squeeze((torch.tensor(self.config['behavior_type'])== torch.tensor(self.config['coarse_grained_type'])[i]).nonzero(), dim = -1))
                if cur_fd_size == 1:
                    continue
                    
                cur_fd = embeddings_each_behavior[torch.squeeze((torch.tensor(self.config['behavior_type']) == torch.tensor(self.config['coarse_grained_type'])[i]).nonzero(), dim = -1)]     # [num_beh_fd, num_users/items, emb_dim]   num_beh_fd: num_beh of this fd
                target_emb = cur_fd[0]  # [num_users/items, emb_dim]
                aux_emb = cur_fd[1:]    # [num_aux, num_users/items, emb_dim]

                batch_target_emb = target_emb[uq_index.to(torch.long),:]                    # [batch_num_users/items, emb_dim]
                batch_aux_emb = aux_emb[:,uq_index.to(torch.long),:]                        # [num_aux, batch_num_users/items, emb_dim]

                if self.config['sample_infonce']:
                    sample_size = int(len(uq_index)*self.config['sample_rate'])
                    sample_index = uq_index[torch.randperm(len(uq_index))[:sample_size].to(torch.long)]
                    
                    sample_target_emb = target_emb[sample_index.to(torch.long),:]
                    sample_aux_emb = aux_emb[:,sample_index.to(torch.long),:]

                    fine_pos_ratings = self.inner_product(sample_target_emb.unsqueeze(0), sample_aux_emb)                      # [num_beh_fd, num_beh_fd, batch_num_users/items]

                    fine_tot_ratings = torch.matmul(sample_target_emb.unsqueeze(0), torch.transpose(sample_aux_emb, -1, -2))   # [num_beh_fd, num_beh_fd, batch_num_users/items, batch_num_users/items]

                    fine_ssl_logits = fine_tot_ratings - fine_pos_ratings.unsqueeze(-1)                                  # [num_beh_fd, num_beh_fd, batch_num_users/items, batch_num_users/items]
                    fine_clogits = torch.logsumexp(fine_ssl_logits / self.config['ssl_temp'], dim=-1)     
                    
                else:
                    if self.config['InfoNCE_loss_batch']:
                        fine_pos_ratings = self.inner_product(batch_target_emb.unsqueeze(0), batch_aux_emb)                      # [num_aux, batch_num_users/items]

                        fine_tot_ratings = torch.matmul(batch_target_emb.unsqueeze(0), torch.transpose(batch_aux_emb, -1, -2))   # [num_aux, batch_num_users/items, batch_num_users/items]

                        fine_ssl_logits = fine_tot_ratings - fine_pos_ratings.unsqueeze(-1)                                      # [num_aux, batch_num_users/items, batch_num_users/items]
                        fine_clogits = torch.logsumexp(fine_ssl_logits / self.config['ssl_temp'], dim=-1)                        # [num_aux, batch_num_users/items]
                
                    else:
                        # calculate InfoNCE_loss between batch nodes and all the other nodes
                        fine_pos_ratings = self.inner_product(batch_target_emb.unsqueeze(0), batch_aux_emb)                            # [num_aux, batch_num_users/items]

                        fine_tot_ratings = torch.matmul(batch_target_emb.unsqueeze(0), torch.transpose(aux_emb, -1, -2))         # [num_aux, batch_num_users/items, num_users/items]

                        fine_ssl_logits = fine_tot_ratings - fine_pos_ratings.unsqueeze(-1)                                            # [num_aux, batch_num_users/items, num_users/items]
                        fine_clogits = torch.logsumexp(fine_ssl_logits / self.config['ssl_temp'], dim=-1)             # [num_aux, batch_num_users/items]    

                fine_grained_InfoNCE_loss += torch.sum(fine_clogits)

        return fine_grained_InfoNCE_loss




    def cal_fine_grained_InfoNCE_loss_paper(self, embeddings_each_behavior, uq_index):
        fine_grained_InfoNCE_loss = torch.tensor(0).to(torch.float).to(self.device)
        
        for i in range(len(torch.tensor(self.config['coarse_grained_type']))):   # EP/0, IP/1, EN/2, IN/3
            if torch.tensor(self.config['coarse_grained_type'])[i] != self.config['target_coarse_id']:
                ## ------------------------ auxiliary behavior one2one -------------------------
                cur_fd_size = len(torch.squeeze((torch.tensor(self.config['behavior_type'])== torch.tensor(self.config['coarse_grained_type'])[i]).nonzero(), dim = -1))
                if cur_fd_size == 1:
                    continue

                cur_fd = embeddings_each_behavior[torch.squeeze((torch.tensor(self.config['behavior_type']) == torch.tensor(self.config['coarse_grained_type'])[i]).nonzero(), dim = -1)]     # [num_beh_fd, num_users/items, emb_dim]   num_beh_fd: num_beh of this fd
                expand_beh = torch.stack([cur_fd for _ in range(cur_fd_size)], dim = 0)    # [num_beh_fd, num_beh_fd, num_users/items, emb_dim]
                
                batch_emb = cur_fd[:,uq_index.to(torch.long),:]                    # [num_beh_fd, batch_num_users/items, emb_dim]
                batch_expand_beh = expand_beh[:,:,uq_index.to(torch.long),:]       # [num_beh_fd, num_beh_fd, batch_num_users/items, emb_dim]

                if self.config['sample_infonce']:
                    sample_size = int(len(uq_index)*self.config['sample_rate'])
                    sample_index = uq_index[torch.randperm(len(uq_index))[:sample_size].to(torch.long)]
                    sample_emb = cur_fd[:,sample_index.to(torch.long),:]
                    expand_sample_emb = expand_beh[:,:,sample_index.to(torch.long),:]

                    fine_pos_ratings = self.inner_product(sample_emb.unsqueeze(1), expand_sample_emb)                      # [num_beh_fd, num_beh_fd, batch_num_users/items]

                    fine_tot_ratings = torch.matmul(sample_emb.unsqueeze(1), torch.transpose(expand_sample_emb, -1, -2))   # [num_beh_fd, num_beh_fd, batch_num_users/items, batch_num_users/items]

                    fine_ssl_logits = fine_tot_ratings - fine_pos_ratings.unsqueeze(-1)                                  # [num_beh_fd, num_beh_fd, batch_num_users/items, batch_num_users/items]
                    fine_clogits = torch.logsumexp(fine_ssl_logits / self.config['ssl_temp'], dim=-1)     
                    
                    # print("aux fine_ssl_logits=",fine_ssl_logits)
                    # if torch.isnan(fine_clogits).any():
                    #     print("nan exist in fine aux")
                    #     input("pause")
                    # if torch.isinf(fine_clogits).any():
                    #     print("inf exist in fine aux")
                    #     input("pause")

                    # fine_pos_ratings = self.inner_product(batch_emb.unsqueeze(1), batch_expand_beh)                      # [num_beh_fd, num_beh_fd, batch_num_users/items]

                    # fine_tot_ratings = torch.matmul(batch_emb.unsqueeze(1), torch.transpose(sample_emb, -1, -2))   # [num_beh_fd, num_beh_fd, batch_num_users/items, batch_num_users/items]

                    # fine_ssl_logits = fine_tot_ratings - fine_pos_ratings.unsqueeze(-1)                                  # [num_beh_fd, num_beh_fd, batch_num_users/items, batch_num_users/items]
                    # fine_clogits = torch.logsumexp(fine_ssl_logits / self.config['ssl_temp'], dim=-1)     

                else:    
                    if self.config['InfoNCE_loss_batch']:
                        fine_pos_ratings = self.inner_product(batch_emb.unsqueeze(1), batch_expand_beh)                      # [num_beh_fd, num_beh_fd, batch_num_users/items]

                        fine_tot_ratings = torch.matmul(batch_emb.unsqueeze(1), torch.transpose(batch_expand_beh, -1, -2))   # [num_beh_fd, num_beh_fd, batch_num_users/items, batch_num_users/items]

                        fine_ssl_logits = fine_tot_ratings - fine_pos_ratings.unsqueeze(-1)                                  # [num_beh_fd, num_beh_fd, batch_num_users/items, batch_num_users/items]
                        fine_clogits = torch.logsumexp(fine_ssl_logits / self.config['ssl_temp'], dim=-1)                                # [num_beh_fd, num_beh_fd, batch_num_users/items]
                
                    else:
                        # calculate InfoNCE_loss between batch nodes and all the other nodes
                        fine_pos_ratings = self.inner_product(batch_emb.unsqueeze(1), batch_expand_beh)                      # [num_beh_fd, num_beh_fd, batch_num_users/items]

                        fine_tot_ratings = torch.matmul(batch_emb.unsqueeze(1), torch.transpose(expand_beh, -1, -2))         # [num_beh_fd, num_beh_fd, batch_num_users/items, num_users/items]

                        fine_ssl_logits = fine_tot_ratings - fine_pos_ratings.unsqueeze(-1)                                  # [num_beh_fd, num_beh_fd, batch_num_users/items, num_users/items]
                        fine_clogits = torch.logsumexp(fine_ssl_logits / self.config['ssl_temp'], dim=-1)                              # [num_beh_fd, num_beh_fd, batch_num_users/items]
            
                InfoNCE_loss_self = torch.sum(fine_clogits, dim = -1)             # [num_beh_fd, num_beh_fd]
                mask_self = torch.eye(cur_fd_size).to(self.device)
                
                fine_InfoNCE_loss = torch.sum(InfoNCE_loss_self * (1-mask_self))      

                fine_grained_InfoNCE_loss += fine_InfoNCE_loss
            
            else:
                ## ------------------------ target behavior -------------------------
                cur_fd_size = len(torch.squeeze((torch.tensor(self.config['behavior_type'])== torch.tensor(self.config['coarse_grained_type'])[i]).nonzero(), dim = -1))
                if cur_fd_size == 1:
                    continue
                    
                cur_fd = embeddings_each_behavior[torch.squeeze((torch.tensor(self.config['behavior_type']) == torch.tensor(self.config['coarse_grained_type'])[i]).nonzero(), dim = -1)]     # [num_beh_fd, num_users/items, emb_dim]   num_beh_fd: num_beh of this fd
                target_emb = cur_fd[0]  # [num_users/items, emb_dim]
                aux_emb = cur_fd[1:]    # [num_aux, num_users/items, emb_dim]

                batch_target_emb = target_emb[uq_index.to(torch.long),:]                    # [batch_num_users/items, emb_dim]
                batch_aux_emb = aux_emb[:,uq_index.to(torch.long),:]                        # [num_aux, batch_num_users/items, emb_dim]

                if self.config['sample_infonce']:
                    sample_size = int(len(uq_index)*self.config['sample_rate'])
                    sample_index = uq_index[torch.randperm(len(uq_index))[:sample_size].to(torch.long)]
                    
                    sample_target_emb = target_emb[sample_index.to(torch.long),:]
                    sample_aux_emb = aux_emb[:,sample_index.to(torch.long),:]

                    fine_pos_ratings = self.inner_product(sample_target_emb.unsqueeze(0), sample_aux_emb)                      # [num_beh_fd, num_beh_fd, batch_num_users/items]

                    fine_tot_ratings = torch.matmul(sample_target_emb.unsqueeze(0), torch.transpose(sample_aux_emb, -1, -2))   # [num_beh_fd, num_beh_fd, batch_num_users/items, batch_num_users/items]

                    fine_ssl_logits = fine_tot_ratings - fine_pos_ratings.unsqueeze(-1)                                  # [num_beh_fd, num_beh_fd, batch_num_users/items, batch_num_users/items]
                    fine_clogits = torch.logsumexp(fine_ssl_logits / self.config['ssl_temp'], dim=-1)     
                    
                    # print("target fine_ssl_logits=",fine_ssl_logits)
                    # if torch.isnan(fine_clogits).any():
                    #     print("exist nan in fine target")
                    #     input("pause")
                    # if torch.isinf(fine_clogits).any():
                    #     print("exist inf in coarse target")
                    #     input("pause")

                    # fine_pos_ratings = self.inner_product(batch_target_emb.unsqueeze(1), sample_emb)                      # [num_beh_fd, num_beh_fd, batch_num_users/items]

                    # fine_tot_ratings = torch.matmul(batch_target_emb.unsqueeze(1), torch.transpose(sample_emb, -1, -2))   # [num_beh_fd, num_beh_fd, batch_num_users/items, batch_num_users/items]

                    # fine_ssl_logits = fine_tot_ratings - fine_pos_ratings.unsqueeze(-1)                                  # [num_beh_fd, num_beh_fd, batch_num_users/items, batch_num_users/items]
                    # fine_clogits = torch.logsumexp(fine_ssl_logits / self.config['ssl_temp'], dim=-1)     
                else:
                    if self.config['InfoNCE_loss_batch']:
                        fine_pos_ratings = self.inner_product(batch_target_emb.unsqueeze(0), batch_aux_emb)                      # [num_aux, batch_num_users/items]

                        fine_tot_ratings = torch.matmul(batch_target_emb.unsqueeze(0), torch.transpose(batch_aux_emb, -1, -2))   # [num_aux, batch_num_users/items, batch_num_users/items]

                        fine_ssl_logits = fine_tot_ratings - fine_pos_ratings.unsqueeze(-1)                                      # [num_aux, batch_num_users/items, batch_num_users/items]
                        fine_clogits = torch.logsumexp(fine_ssl_logits / self.config['ssl_temp'], dim=-1)                        # [num_aux, batch_num_users/items]
                
                    else:
                        # calculate InfoNCE_loss between batch nodes and all the other nodes
                        fine_pos_ratings = self.inner_product(batch_target_emb.unsqueeze(0), batch_aux_emb)                            # [num_aux, batch_num_users/items]

                        fine_tot_ratings = torch.matmul(batch_target_emb.unsqueeze(0), torch.transpose(aux_emb, -1, -2))         # [num_aux, batch_num_users/items, num_users/items]

                        fine_ssl_logits = fine_tot_ratings - fine_pos_ratings.unsqueeze(-1)                                            # [num_aux, batch_num_users/items, num_users/items]
                        fine_clogits = torch.logsumexp(fine_ssl_logits / self.config['ssl_temp'], dim=-1)             # [num_aux, batch_num_users/items]    

                fine_grained_InfoNCE_loss += torch.sum(fine_clogits)

        return fine_grained_InfoNCE_loss


    def cal_coarse_grained_InfoNCE_loss_paper_wo_sample(self, embeddings_each_feedback, uq_index, node_type):
        """
        conduct contrastive learning within EP/IP and EN/IN 
        """
        #print("len_uq_index",len(uq_index))
        uq_index = torch.as_tensor(np.random.choice(uq_index.cpu(), size=int(len(uq_index)/10), replace=False, p=None)).to(self.config['device'])
        infonce_fold = self.config['infonce_fold']
        coarse_InfoNCE_loss = torch.tensor(0).to(torch.float).to(self.device)
        if len(torch.tensor(self.config['coarse_grained_type'])) <= 1:
            return coarse_InfoNCE_loss

        if 0 in torch.tensor(self.config['coarse_grained_type']) and 1 in torch.tensor(self.config['coarse_grained_type']): 
            # EP, IP
            pos_names = []
            for beh_index in range(len(self.config['behavior_name'])):  
                if self.config['behavior_type'][beh_index] <= 1:    
                    pos_names.append(self.config['behavior_name'][beh_index])
            beh_names = ''
            for i in range(len(pos_names)):
                if i != len(pos_names)-1:
                        beh_names += pos_names[i]+'_'
                else:
                    beh_names += pos_names[i]

            inter_file = self.config['data_dir'] + beh_names + '_' + node_type+'.txt'
            inter_node_set = np.loadtxt(inter_file, dtype=int)

            np_uq_index = uq_index.cpu().numpy()
            tri_inter = np.intersect1d(np_uq_index, inter_node_set,  assume_unique=True)

            tri_inter = torch.from_numpy(tri_inter)    

            EP_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type'])==0).nonzero(), dim = -1)
            EP_embeddings_batch = embeddings_each_feedback[EP_index.to(torch.long), tri_inter.to(torch.long)]     # [batch_size, emb_dim]

            IP_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type'])==1).nonzero(), dim = -1)
            IP_embeddings_batch = embeddings_each_feedback[IP_index.to(torch.long), tri_inter.to(torch.long)]     # [batch_size, emb_dim]

            fold_len = len(tri_inter) // infonce_fold
            for i_fold in range(int(infonce_fold)):
                #print("i_fold=",i_fold)
                start = i_fold * fold_len
                if i_fold == infonce_fold - 1:
                    end = len(tri_inter)
                else:
                    end = (i_fold+1)*fold_len

                EP_emb_fold = EP_embeddings_batch[start:end]    
                IP_emb_fold = IP_embeddings_batch[start:end]   

                coarse_pos_ratings_EP_IP = self.inner_product(EP_emb_fold, IP_emb_fold)               
                coarse_tot_ratings_EP_IP = torch.matmul(EP_emb_fold, torch.transpose(IP_embeddings_batch, -1, -2))     
                coarse_ssl_logits_EP_IP = coarse_tot_ratings_EP_IP - coarse_pos_ratings_EP_IP.unsqueeze(-1)  
                coarse_clogits_EP_IP = torch.logsumexp(coarse_ssl_logits_EP_IP / self.config['ssl_temp'], dim=-1)   


                coarse_pos_ratings_IP_EP = self.inner_product(IP_emb_fold, EP_emb_fold)               
                coarse_tot_ratings_IP_EP = torch.matmul(IP_emb_fold, torch.transpose(EP_embeddings_batch, -1, -2))     
                coarse_ssl_logits_IP_EP = coarse_tot_ratings_IP_EP - coarse_pos_ratings_IP_EP.unsqueeze(-1)  
                coarse_clogits_IP_EP = torch.logsumexp(coarse_ssl_logits_IP_EP / self.config['ssl_temp'], dim=-1)  

            
                if self.config['target_coarse_id'] != 0 and self.config['target_coarse_id'] != 1:
                    coarse_InfoNCE_loss += (torch.sum(coarse_clogits_EP_IP)+torch.sum(coarse_clogits_IP_EP))
                elif self.config['target_coarse_id'] == 0:
                    coarse_InfoNCE_loss += torch.sum(coarse_clogits_EP_IP)
                elif self.config['target_coarse_id'] == 1:
                    coarse_InfoNCE_loss += torch.sum(coarse_clogits_IP_EP)
                
             
        if 2 in torch.tensor(self.config['coarse_grained_type']) and 3 in torch.tensor(self.config['coarse_grained_type']):
            # EN, IN
            neg_names = []
            for beh_index in range(len(self.config['behavior_name'])):  
                if self.config['behavior_type'][beh_index] > 1:    # 负向行为
                    neg_names.append(self.config['behavior_name'][beh_index])
            beh_names = ''
            for i in range(len(neg_names)):
                if i != len(neg_names)-1:
                    beh_names += neg_names[i]+'_'
                else:
                    beh_names += neg_names[i]

            inter_file = self.config['data_dir'] + beh_names + '_' + node_type+'.txt'
            inter_node_set = np.loadtxt(inter_file, dtype=int)

            np_uq_index = uq_index.cpu().numpy()
            tri_inter = np.intersect1d(np_uq_index, inter_node_set,  assume_unique=True)

            tri_inter = torch.from_numpy(tri_inter)
            
            EN_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type'])==2).nonzero(), dim = -1)
            EN_embeddings_batch = embeddings_each_feedback[EN_index.to(torch.long), tri_inter.to(torch.long)]     # [batch_size, emb_dim]
                
            IN_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type'])==3).nonzero(), dim = -1)
            IN_embeddings_batch = embeddings_each_feedback[IN_index.to(torch.long), tri_inter.to(torch.long)]     # [batch_size, emb_dim]

            fold_len = len(tri_inter) // infonce_fold
            for i_fold in range(int(infonce_fold)):
                start = i_fold * fold_len
                if i_fold == infonce_fold - 1:
                    end = len(tri_inter)
                else:
                    end = (i_fold+1)*fold_len

                EN_emb_fold = EN_embeddings_batch[start:end]    
                IN_emb_fold = IN_embeddings_batch[start:end]   

                coarse_pos_ratings_EN_IN = self.inner_product(EN_emb_fold, IN_emb_fold)               
                coarse_tot_ratings_EN_IN = torch.matmul(EN_emb_fold, torch.transpose(IN_embeddings_batch, -1, -2))     
                coarse_ssl_logits_EN_IN = coarse_tot_ratings_EN_IN - coarse_pos_ratings_EN_IN.unsqueeze(-1)  
                coarse_clogits_EN_IN = torch.logsumexp(coarse_ssl_logits_EN_IN / self.config['ssl_temp'], dim=-1)   


                coarse_pos_ratings_IN_EN = self.inner_product(IN_emb_fold, EN_emb_fold)               
                coarse_tot_ratings_IN_EN = torch.matmul(IN_emb_fold, torch.transpose(EN_embeddings_batch, -1, -2))     
                coarse_ssl_logits_IN_EN = coarse_tot_ratings_IN_EN - coarse_pos_ratings_IN_EN.unsqueeze(-1)  
                coarse_clogits_IN_EN = torch.logsumexp(coarse_ssl_logits_IN_EN / self.config['ssl_temp'], dim=-1)  

            
                if self.config['target_coarse_id'] != 2 and self.config['target_coarse_id'] != 3:
                    coarse_InfoNCE_loss += (torch.sum(coarse_clogits_EN_IN)+torch.sum(coarse_clogits_IN_EN))
                elif self.config['target_coarse_id'] == 2:
                    coarse_InfoNCE_loss += torch.sum(coarse_clogits_EN_IN)
                elif self.config['target_coarse_id'] == 3:
                    coarse_InfoNCE_loss += torch.sum(coarse_clogits_IN_EN)

        return coarse_InfoNCE_loss  




    def cal_coarse_grained_InfoNCE_loss_paper(self, embeddings_each_feedback, uq_index):
            """
            conduct contrastive learning within EP/IP and EN/IN 
            """
            
            coarse_InfoNCE_loss = torch.tensor(0).to(torch.float).to(self.device)
            if len(torch.tensor(self.config['coarse_grained_type'])) <= 1:
                return coarse_InfoNCE_loss

            if self.config['sample_infonce']:
                sample_size = int(len(uq_index)*self.config['sample_rate'])
                sample_index = uq_index[torch.randperm(len(uq_index))[:sample_size].to(torch.long)]

                if 0 in torch.tensor(self.config['coarse_grained_type']) and 1 in torch.tensor(self.config['coarse_grained_type']): 
                    # EP, IP
                    EP_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type'])==0).nonzero(), dim = -1)
                    EP_embeddings_batch = embeddings_each_feedback[EP_index.to(torch.long), uq_index.to(torch.long)]     # [batch_size, emb_dim]
                    EP_embeddings_sample = embeddings_each_feedback[EP_index.to(torch.long), sample_index.to(torch.long)]     # [batch_size, emb_dim]

                    IP_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type'])==1).nonzero(), dim = -1)
                    IP_embeddings_batch = embeddings_each_feedback[IP_index.to(torch.long), uq_index.to(torch.long)]     # [batch_size, emb_dim]
                    IP_embeddings_sample = embeddings_each_feedback[IP_index.to(torch.long), sample_index.to(torch.long)]     # [batch_size, emb_dim]
                    
                    coarse_pos_ratings_EP_IP = self.inner_product(EP_embeddings_sample, IP_embeddings_sample)               # [batch_size]
                    coarse_tot_ratings_EP_IP = torch.matmul(EP_embeddings_sample, torch.transpose(IP_embeddings_sample, -1,-2))                     # [batch_size, batch_size]
                    coarse_ssl_logits_EP_IP = coarse_tot_ratings_EP_IP - coarse_pos_ratings_EP_IP.unsqueeze(-1)           # [batch_size, batch_size]
                    coarse_clogits_EP_IP = torch.logsumexp(coarse_ssl_logits_EP_IP / self.config['ssl_temp'], dim=-1)     # [batch_size]
                    # print("coarse_ssl_logits_EP_IP=",coarse_ssl_logits_EP_IP)
                    # if torch.isnan(coarse_clogits_EP_IP).any():
                    #     print("exist nan in coarse P")
                    #     input("pause")
                    # if torch.isinf(coarse_clogits_EP_IP).any():
                    #     print("exist inf in coarse P")
                    #     input("pause")                    

                    coarse_pos_ratings_IP_EP = self.inner_product(IP_embeddings_sample, EP_embeddings_sample)               # [batch_size]
                    coarse_tot_ratings_IP_EP = torch.matmul(IP_embeddings_sample, torch.transpose(EP_embeddings_sample, -1,-2))                     # [batch_size, batch_size]
                    coarse_ssl_logits_IP_EP = coarse_tot_ratings_IP_EP - coarse_pos_ratings_IP_EP.unsqueeze(-1)           # [batch_size, batch_size]
                    coarse_clogits_IP_EP = torch.logsumexp(coarse_ssl_logits_IP_EP / self.config['ssl_temp'], dim=-1)     # [batch_size]


                    if self.config['target_coarse_id'] != 0 and self.config['target_coarse_id'] != 1:
                        coarse_InfoNCE_loss += (torch.sum(coarse_clogits_EP_IP)+torch.sum(coarse_clogits_IP_EP))
                    elif self.config['target_coarse_id'] == 0:
                        coarse_InfoNCE_loss += torch.sum(coarse_clogits_EP_IP)
                    elif self.config['target_coarse_id'] == 1:
                        coarse_InfoNCE_loss += torch.sum(coarse_clogits_IP_EP)

                if 2 in torch.tensor(self.config['coarse_grained_type']) and 3 in torch.tensor(self.config['coarse_grained_type']):
                    # EN, IN
                    EN_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type'])==2).nonzero(), dim = -1)
                    EN_embeddings_batch = embeddings_each_feedback[EN_index.to(torch.long), uq_index.to(torch.long)]     # [batch_size, emb_dim]
                    EN_embeddings_sample = embeddings_each_feedback[EN_index.to(torch.long), sample_index.to(torch.long)]     # [batch_size, emb_dim]

                    IN_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type'])==3).nonzero(), dim = -1)
                    IN_embeddings_batch = embeddings_each_feedback[IN_index.to(torch.long), uq_index.to(torch.long)]     # [batch_size, emb_dim]
                    IN_embeddings_sample = embeddings_each_feedback[IN_index.to(torch.long), sample_index.to(torch.long)]     # [batch_size, emb_dim]

                    coarse_pos_ratings_EN_IN = self.inner_product(EN_embeddings_sample, IN_embeddings_sample)               # [batch_size]
                    coarse_tot_ratings_EN_IN = torch.matmul(EN_embeddings_sample, torch.transpose(IN_embeddings_sample,-1,-2))                     # [batch_size, batch_size]
                    coarse_ssl_logits_EN_IN = coarse_tot_ratings_EN_IN - coarse_pos_ratings_EN_IN.unsqueeze(-1)           # [batch_size, batch_size]
                    coarse_clogits_EN_IN = torch.logsumexp(coarse_ssl_logits_EN_IN / self.config['ssl_temp'], dim=-1)     # [batch_size]
                    # print("coarse_ssl_logits_EN_IN=",coarse_ssl_logits_EN_IN)
                    # if torch.isnan(coarse_clogits_EN_IN).any():
                    #     print("exist nan in coarse N")
                    #     input("pause")
                    # if torch.isinf(coarse_clogits_EN_IN).any():
                    #     print("exist inf in coarse N")
                    #     input("pause")
                    
                    coarse_pos_ratings_IN_EN = self.inner_product(IN_embeddings_sample, EN_embeddings_sample)               # [batch_size]
                    coarse_tot_ratings_IN_EN = torch.matmul(IN_embeddings_sample, torch.transpose(EN_embeddings_sample,-1,-2))                     # [batch_size, batch_size]
                    coarse_ssl_logits_IN_EN = coarse_tot_ratings_IN_EN - coarse_pos_ratings_IN_EN.unsqueeze(-1)           # [batch_size, batch_size]
                    coarse_clogits_IN_EN = torch.logsumexp(coarse_ssl_logits_IN_EN / self.config['ssl_temp'], dim=-1)     # [batch_size]
            
                    if self.config['target_coarse_id'] != 2 and self.config['target_coarse_id'] != 3:
                        coarse_InfoNCE_loss += (torch.sum(coarse_clogits_EN_IN)+torch.sum(coarse_clogits_IN_EN))
                    elif self.config['target_coarse_id'] == 2:
                        coarse_InfoNCE_loss += torch.sum(coarse_clogits_EN_IN)
                    elif self.config['target_coarse_id'] == 3:
                        coarse_InfoNCE_loss += torch.sum(coarse_clogits_IN_EN)
            else:
                if self.config['InfoNCE_loss_batch']:
                    if 0 in torch.tensor(self.config['coarse_grained_type']) and 1 in torch.tensor(self.config['coarse_grained_type']): 
                        # EP, IP
                        EP_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type'])==0).nonzero(), dim = -1)
                        EP_embeddings_batch = embeddings_each_feedback[EP_index.to(torch.long), uq_index.to(torch.long)]     # [batch_size, emb_dim]

                        IP_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type'])==1).nonzero(), dim = -1)
                        IP_embeddings_batch = embeddings_each_feedback[IP_index.to(torch.long), uq_index.to(torch.long)]     # [batch_size, emb_dim]
                        
                        coarse_pos_ratings_EP_IP = self.inner_product(EP_embeddings_batch, IP_embeddings_batch)               # [batch_size]
                        coarse_tot_ratings_EP_IP = torch.matmul(EP_embeddings_batch, torch.transpose(IP_embeddings_batch, -1,-2))                     # [batch_size, batch_size]
                        coarse_ssl_logits_EP_IP = coarse_tot_ratings_EP_IP - coarse_pos_ratings_EP_IP.unsqueeze(-1)           # [batch_size, batch_size]
                        coarse_clogits_EP_IP = torch.logsumexp(coarse_ssl_logits_EP_IP / self.config['ssl_temp'], dim=-1)     # [batch_size]

                        coarse_pos_ratings_IP_EP = self.inner_product(IP_embeddings_batch, EP_embeddings_batch)               # [batch_size]
                        coarse_tot_ratings_IP_EP = torch.matmul(IP_embeddings_batch, torch.transpose(EP_embeddings_batch, -1,-2))                     # [batch_size, batch_size]
                        coarse_ssl_logits_IP_EP = coarse_tot_ratings_IP_EP - coarse_pos_ratings_IP_EP.unsqueeze(-1)           # [batch_size, batch_size]
                        coarse_clogits_IP_EP = torch.logsumexp(coarse_ssl_logits_IP_EP / self.config['ssl_temp'], dim=-1)     # [batch_size]

                        if self.config['target_coarse_id'] != 0 and self.config['target_coarse_id'] != 1:
                            coarse_InfoNCE_loss += (torch.sum(coarse_clogits_EP_IP)+torch.sum(coarse_clogits_IP_EP))
                        elif self.config['target_coarse_id'] == 0:
                            coarse_InfoNCE_loss += torch.sum(coarse_clogits_EP_IP)
                        elif self.config['target_coarse_id'] == 1:
                            coarse_InfoNCE_loss += torch.sum(coarse_clogits_IP_EP)

                    if 2 in torch.tensor(self.config['coarse_grained_type']) and 3 in torch.tensor(self.config['coarse_grained_type']):
                        # EN, IN
                        EN_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type'])==2).nonzero(), dim = -1)
                        EN_embeddings_batch = embeddings_each_feedback[EN_index.to(torch.long), uq_index.to(torch.long)]     # [batch_size, emb_dim]

                        IN_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type'])==3).nonzero(), dim = -1)
                        IN_embeddings_batch = embeddings_each_feedback[IN_index.to(torch.long), uq_index.to(torch.long)]     # [batch_size, emb_dim]

                        coarse_pos_ratings_EN_IN = self.inner_product(EN_embeddings_batch, IN_embeddings_batch)               # [batch_size]
                        coarse_tot_ratings_EN_IN = torch.matmul(EN_embeddings_batch, torch.transpose(IN_embeddings_batch,-1,-2))                     # [batch_size, batch_size]
                        coarse_ssl_logits_EN_IN = coarse_tot_ratings_EN_IN - coarse_pos_ratings_EN_IN.unsqueeze(-1)           # [batch_size, batch_size]
                        coarse_clogits_EN_IN = torch.logsumexp(coarse_ssl_logits_EN_IN / self.config['ssl_temp'], dim=-1)     # [batch_size]

                        coarse_pos_ratings_IN_EN = self.inner_product(IN_embeddings_batch, EN_embeddings_batch)               # [batch_size]
                        coarse_tot_ratings_IN_EN = torch.matmul(IN_embeddings_batch, torch.transpose(EN_embeddings_batch,-1,-2))                     # [batch_size, batch_size]
                        coarse_ssl_logits_IN_EN = coarse_tot_ratings_IN_EN - coarse_pos_ratings_IN_EN.unsqueeze(-1)           # [batch_size, batch_size]
                        coarse_clogits_IN_EN = torch.logsumexp(coarse_ssl_logits_IN_EN / self.config['ssl_temp'], dim=-1)     # [batch_size]
                
                        if self.config['target_coarse_id'] != 2 and self.config['target_coarse_id'] != 3:
                            coarse_InfoNCE_loss += (torch.sum(coarse_clogits_EN_IN)+torch.sum(coarse_clogits_IN_EN))/2
                        elif self.config['target_coarse_id'] == 2:
                            coarse_InfoNCE_loss += torch.sum(coarse_clogits_EN_IN)/2
                        elif self.config['target_coarse_id'] == 3:
                            coarse_InfoNCE_loss += torch.sum(coarse_clogits_IN_EN)/2

                else:
                    if 0 in torch.tensor(self.config['coarse_grained_type']) and 1 in torch.tensor(self.config['coarse_grained_type']): 
                        # EP, IP
                        EP_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type'])==0).nonzero(), dim = -1)
                        EP_embeddings_batch = embeddings_each_feedback[EP_index.to(torch.long), uq_index.to(torch.long)]     # [batch_size, emb_dim]
                        EP_embeddings = embeddings_each_feedback[EP_index.to(torch.long)]     # [num_users/items, emb_dim]

                        IP_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type'])==1).nonzero(), dim = -1)
                        IP_embeddings_batch = embeddings_each_feedback[IP_index.to(torch.long), uq_index.to(torch.long)]     # [batch_size, emb_dim]
                        IP_embeddings = embeddings_each_feedback[IP_index.to(torch.long)]     # [num_users/items, emb_dim]

                        coarse_pos_ratings_EP_IP = self.inner_product(EP_embeddings_batch, IP_embeddings_batch)               # [batch_size]
                        coarse_tot_ratings_EP_IP = torch.matmul(EP_embeddings_batch, torch.transpose(IP_embeddings, -1, -2))                           # [batch_size, num_users/items]
                        coarse_ssl_logits_EP_IP = coarse_tot_ratings_EP_IP - coarse_pos_ratings_EP_IP.unsqueeze(-1)           # [batch_size, num_users/items]
                        coarse_clogits_EP_IP = torch.logsumexp(coarse_ssl_logits_EP_IP / self.config['ssl_temp'], dim=-1)     # [batch_size]
                        loss_ep_ip = torch.sum(coarse_clogits_EP_IP)

                        coarse_pos_ratings_IP_EP = self.inner_product(IP_embeddings_batch, EP_embeddings_batch)               # [batch_size]
                        coarse_tot_ratings_IP_EP = torch.matmul(IP_embeddings_batch, torch.transpose(EP_embeddings,-1,-2))                           # [batch_size, num_users/items]
                        coarse_ssl_logits_IP_EP = coarse_tot_ratings_IP_EP - coarse_pos_ratings_IP_EP.unsqueeze(-1)           # [batch_size, num_users/items]
                        coarse_clogits_IP_EP = torch.logsumexp(coarse_ssl_logits_IP_EP / self.config['ssl_temp'], dim=-1)     # [batch_size]
                        loss_ip_ep = torch.sum(coarse_clogits_IP_EP)

                        if self.config['target_coarse_id'] != 0 and self.config['target_coarse_id'] != 1:
                            coarse_InfoNCE_loss += (loss_ep_ip + loss_ip_ep)/2
                        elif self.config['target_coarse_id'] == 0:
                            coarse_InfoNCE_loss += loss_ep_ip/2
                        elif self.config['target_coarse_id'] == 1:
                            coarse_InfoNCE_loss += loss_ip_ep/2

                    if 2 in torch.tensor(self.config['coarse_grained_type']) and 3 in torch.tensor(self.config['coarse_grained_type']):
                        # EN, IN
                        EN_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type'])==2).nonzero(), dim = -1)
                        EN_embeddings_batch = embeddings_each_feedback[EN_index.to(torch.long), uq_index.to(torch.long)]     # [batch_size, emb_dim]
                        EN_embeddings = embeddings_each_feedback[EN_index.to(torch.long)]     # [num_users/items, emb_dim]

                        IN_index = torch.squeeze((torch.tensor(self.config['coarse_grained_type'])==3).nonzero(), dim = -1)
                        IN_embeddings_batch = embeddings_each_feedback[IN_index.to(torch.long), uq_index.to(torch.long)]     # [batch_size, emb_dim]
                        IN_embeddings = embeddings_each_feedback[IN_index.to(torch.long)]     # [num_users/items, emb_dim]

                        coarse_pos_ratings_EN_IN = self.inner_product(EN_embeddings_batch, IN_embeddings_batch)               # [batch_size]
                        coarse_tot_ratings_EN_IN = torch.matmul(EN_embeddings_batch, torch.transpose(IN_embeddings, -1, -2))                           # [batch_size, num_users/items]
                        coarse_ssl_logits_EN_IN = coarse_tot_ratings_EN_IN - coarse_pos_ratings_EN_IN.unsqueeze(-1)           # [batch_size, num_users/items]
                        coarse_clogits_EN_IN = torch.logsumexp(coarse_ssl_logits_EN_IN / self.config['ssl_temp'], dim=-1)     # [batch_size]
                        loss_en_in = torch.sum(coarse_clogits_EN_IN)

                        coarse_pos_ratings_IN_EN = self.inner_product(IN_embeddings_batch, EN_embeddings_batch)               # [batch_size]
                        coarse_tot_ratings_IN_EN = torch.matmul(IN_embeddings_batch, torch.transpose(EN_embeddings, -1, -2))                           # [batch_size, num_users/items]
                        coarse_ssl_logits_IN_EN = coarse_tot_ratings_IN_EN - coarse_pos_ratings_IN_EN.unsqueeze(-1)           # [batch_size, num_users/items]
                        coarse_clogits_IN_EN = torch.logsumexp(coarse_ssl_logits_IN_EN / self.config['ssl_temp'], dim=-1)     # [batch_size]
                        loss_in_en = torch.sum(coarse_clogits_IN_EN)

                        if self.config['target_coarse_id'] != 2 and self.config['target_coarse_id'] != 3:
                            coarse_InfoNCE_loss += (loss_en_in + loss_in_en)
                        elif self.config['target_coarse_id'] == 2:
                            coarse_InfoNCE_loss += loss_en_in
                        elif self.config['target_coarse_id'] == 3:
                            coarse_InfoNCE_loss += loss_in_en

            return coarse_InfoNCE_loss   
   
   

    def cal_user_topk(self, user, user_embedding, item_embedding, coarse_grained_type, dim_each_feedback, \
        dot_weights, score_choice, target_behavior_id, all_rank_flag, is_testing, test_user_item):
        """
        calculate topk item index for user in current batch
        params:
            user_embedding: learned user embedding,  [batch_size, emb_dim]  
            item_embedding: learned item embedding,  [num_items, emb_dim]  
            coarse_grained_type: list of int, e.g. [0, 1, 2, 3], [0,1]   0-EP, 1-IP, 2-EN, 3-IN    default: positive before negative
            dot_weights: list of float, e.g. [1.0, 1.0, 1.0, 1.0]
            score_choice: decide how to calculate score: dot, dot_minus
            target_behavior_id: id of test behavior
        return:
            user_topk  [batch_size, topk]
        """

        if all_rank_flag:
            cur_user_embedding = user_embedding[user]
            if score_choice == 'dot':
                score_matrix = torch.matmul(cur_user_embedding, torch.transpose(item_embedding, -1, -2)) 
            else:
                score_positive = torch.zeros((cur_user_embedding.shape[0], item_embedding.shape[0]))
                score_negative = torch.zeros((cur_user_embedding.shape[0], item_embedding.shape[0]))
                for index, type in enumerate(coarse_grained_type):
                    if type <= 1:  # positive feedback
                        score_positive += dot_weights[index]*torch.matmul(cur_user_embedding[:, index*dim_each_feedback : (index + 1)*dim_each_feedback], \
                            torch.transpose(item_embedding[:, index*dim_each_feedback : (index + 1)*dim_each_feedback], -1, -2))
                    else: 
                        score_negative += dot_weights[index]*torch.matmul(cur_user_embedding[:, index*dim_each_feedback : (index + 1)*dim_each_feedback], \
                            torch.transpose(item_embedding[:, index*dim_each_feedback : (index + 1)*dim_each_feedback], -1, -2))

                score_matrix = score_positive - score_negative

            # set the score of target items in training set to -INF
            # to avoid retrivial them in testing process
            # positive_index = torch.squeeze((torch.from_numpy(self.train_loader.dataset.multi_beh_id) == target_behavior_id).nonzero(), dim = -1)
            cur_user_inter = self.train_loader.dataset.multi_beh_data_sp_matrix[target_behavior_id]
            for index, val in enumerate(user):
                user_line = cur_user_inter[val]._indices()[0]
                score_matrix[index, user_line] = -torch._six.inf
               
            # score_matrix[indices[0,:], indices[1,:]] = -torch._six.inf

            if is_testing:
                valid_data = self.valid_loader.dataset.user_item_data_test
                for index, val in enumerate(user):
                    cur_user_index = np.where(valid_data[:,0]==val)[0]
                    score_matrix[index, valid_data[cur_user_index][:,1]] = -torch._six.inf

            _, user_topk = torch.topk(score_matrix, self.config['topk'][-1], dim = -1) 

        else:
            cur_user_embedding = user_embedding[user]
            temp_len = cur_user_embedding.shape[0] * 100
            user_compute = -1 * torch.ones(temp_len)
            item_compute = -1 * torch.ones(temp_len)
            # temp = self.train_loader.dataset.multi_beh_data_sp_matrix[target_behavior_id].to_dense()
            temp = self.train_loader.dataset.target_behavior_dense_matrix
            if is_testing:
            
                valid_data = self.valid_loader.dataset.user_item_data_test
                temp[valid_data[:,0],valid_data[:,1]] = 1

            for i in range(cur_user_embedding.shape[0]):
                candidate_set = torch.squeeze((temp[user[i]] == 0).nonzero(), dim = -1)
                pos_index = torch.squeeze((test_user_item[i]==1).nonzero(), dim = -1)
                random_neg_sam = candidate_set[torch.randperm(len(candidate_set))][:100]  
                user_compute[i*100:(i+1)*100] = user[i] 
                # if len(pos_index) == 1:
                item_compute[i*100:(i+1)*100-1] = random_neg_sam[:99]
                item_compute[(i+1)*100-1] = pos_index
                # else:
                #     item_compute[i*100:(i+1)*100] = random_neg_sam
            
            cal_user_embd = user_embedding[user_compute.to(torch.long)]
            cal_item_embd = item_embedding[item_compute.to(torch.long)]
            if score_choice == 'dot':
                score_matrix = torch.sum(torch.mul(cal_user_embd, cal_item_embd), dim = -1)      # [100 * batch, 1]
            else:
                score_positive = torch.zeros(cal_user_embd.shape[0]).to(self.device)
                score_negative = torch.zeros(cal_user_embd.shape[0]).to(self.device)
                # print(cal_user_embd.device)
                # print(cal_item_embd.device)
                for index, type in enumerate(coarse_grained_type):
                    if type <= 1:  # positive feedback
                        score_positive += dot_weights[index]*torch.sum(torch.mul(cal_user_embd[:, index*dim_each_feedback : (index + 1)*dim_each_feedback], \
                            cal_item_embd[:, index*dim_each_feedback : (index + 1)*dim_each_feedback]), dim = -1)
                    else: 
                        score_negative += dot_weights[index]*torch.sum(torch.mul(cal_user_embd[:, index*dim_each_feedback : (index + 1)*dim_each_feedback], \
                            cal_item_embd[:, index*dim_each_feedback : (index + 1)*dim_each_feedback]), dim = -1)
                score_matrix = score_positive - score_negative
            
            score_matrix = score_matrix.reshape((len(user),-1))
            _, user_topk = torch.topk(score_matrix, self.config['topk'][-1], dim = -1) 
            
        
            row_index = torch.unsqueeze(torch.arange(0, user_topk.shape[0]), dim = -1)
            row_index = row_index.repeat(1, user_topk.shape[1]).reshape((-1,1))
            col_index = user_topk.reshape((-1,1))
            orig_item = item_compute.reshape((len(user),-1))
            user_topk = orig_item[row_index[:,0].to(torch.long), col_index[:,0].to(torch.long)].reshape((len(user), -1)).to(torch.long)

            item_compute = item_compute.reshape((len(user),-1))

        return user_topk, item_compute

        
    
    # ------------------------------------------ Metric ------------------------------------------

    def cal_AUC_metric(self, y_true, y_pred):
        score = roc_auc_score(y_true, y_pred)
        return score


    def cal_HitRatio_metric(self, user_topk, user_item_data):
        """
        params:
            user_topk: [test_batch_size, topk]      topk items for each user in batch
            user_item_data: [test_batch_size, num_items]       groundtruth interaction data
        """
        hit_cnt_list = []
        for topk in self.config['topk']:
            topk_matrix = torch.zeros((user_topk.shape[0], user_item_data.shape[1]))
            row_index = torch.unsqueeze(torch.arange(0,user_topk.shape[0]), dim=-1)
            row_index = row_index.repeat(1,topk).reshape((-1,1))
            col_index = user_topk[:,:topk].reshape((-1,1))
            topk_matrix[row_index[:,0].to(torch.long), col_index[:,0].to(torch.long)] = 1

            hit_cnt = torch.sum(topk_matrix * user_item_data).numpy()
            hit_cnt_list.append(hit_cnt)

        return hit_cnt_list
    

    def cal_NDCG_metric(self, user_topk, user_item_data, pre_cal_IDCG):
        """
        params:
            user_topk: [test_batch_size, topk]      topk items for each user in batch
            user_item_data: [test_batch_size, num_items]       groundtruth interaction data
            pre_cal_IDCG: pre-calculated IDCG
        """
        NDCG_list = []
        for topk in self.config['topk']:
            # calculate DCG        
            row_index = torch.unsqueeze(torch.arange(0,user_topk.shape[0]), dim=-1)
            row_index = row_index.repeat(1,topk).reshape((-1,1))
            col_index = user_topk[:, :topk].reshape((-1,1))
            topk_rel_matrix = user_item_data[row_index[:,0].to(torch.long), col_index[:,0].to(torch.long)].reshape((user_topk.shape[0], topk))     # [batch_size, topk], 0/1  0: false 1:true
            pos = torch.arange(1, topk+1)
            DCG = torch.sum(topk_rel_matrix / torch.log2((torch.unsqueeze(pos, dim = 0) + 1)), dim = -1)

            # calculate IDCG
            positive_cnt = torch.sum(user_item_data, dim = -1)      # [batch_size]
            positive_cnt[torch.squeeze((positive_cnt>=topk).nonzero(), dim = -1)] = topk
            IDCG = pre_cal_IDCG[positive_cnt.to(torch.long)]+ 1e-8

            # NDCG
            NDCG = torch.sum(DCG / IDCG).numpy()
            NDCG_list.append(NDCG)

        return NDCG_list


    def cal_pre_IDCG(self, max_cnt):
        index = torch.arange(0, max_cnt)
        temp = 1.0 / torch.log2(index + 2)
        IDCG = torch.zeros(max_cnt)
        IDCG[0] = temp[0]
        for i in range(max_cnt - 1):
            IDCG[i+1] = IDCG[i] + temp[i+1]
        
        self.pre_cal_IDCG = IDCG
