import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from EduCDM import CDM
import time
import os




class DynamicNodeMask(nn.Module):
    def __init__(self, base_mask_ratio, embedding_size):
        super(DynamicNodeMask, self).__init__()
        self.base_mask_ratio = base_mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, embedding_size))
    
    def forward(self, embeds, dynamic_factors):
        batch_size, embedding_size = embeds.shape
        masked_embeds = embeds.clone()
        all_seeds = []
        
        for i in range(batch_size):
            mask_ratio = self.base_mask_ratio * dynamic_factors[i] 
            num_masked = max(int(embedding_size * mask_ratio.item()), 1)
            seeds = np.random.choice(embedding_size, size=num_masked, replace=False)
            seeds = torch.LongTensor(seeds).to(embeds.device)
            mask = torch.ones(embedding_size).to(embeds.device)
            mask[seeds] = 0
            masked_embeds[i] = embeds[i] * mask + self.mask_token * (1. - mask)
            all_seeds.append(seeds)
        
        return masked_embeds, all_seeds

def combine_tensors(tensor_list, target_length):
    padded_tensors = []
    for tensor in tensor_list:
        
        padding_length = target_length - tensor.size(0)
       
        if padding_length > 0:
            padded_tensor = F.pad(tensor, (0, padding_length), "constant", 0)
            padded_tensors.append(padded_tensor)
        else:
            padded_tensors.append(tensor)
    
   
    combined_tensor = torch.stack(padded_tensors)
    return combined_tensor

def cal_infonce_loss(embeds1, embeds2, all_embeds2, temp=1.0):
    normed_embeds1 = embeds1 / torch.sqrt(1e-8 + embeds1.square().sum(-1, keepdim=True))
    normed_embeds2 = embeds2 / torch.sqrt(1e-8 + embeds2.square().sum(-1, keepdim=True))
    normed_all_embeds2 = all_embeds2 / torch.sqrt(1e-8 + all_embeds2.square().sum(-1, keepdim=True))
    nume_term = torch.exp(normed_embeds1 * normed_embeds2 / temp).sum()
    deno_term = torch.sum(torch.exp(normed_embeds1 @ normed_all_embeds2.T / temp), dim=-1)
    cl_loss = -torch.log(nume_term / (deno_term + 1e-8) + 1e-8).mean()
    return cl_loss

def ssl_con_loss(x, y, temp=1.0):
    x = F.normalize(x)
    y = F.normalize(y)
    mole = torch.exp(torch.sum(x * y, dim=1) / temp)
    deno = torch.sum(torch.exp(x @ y.T / temp), dim=1)
    return -torch.log(mole / (deno + 1e-8) + 1e-8).mean()

class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)

class NCDNet(nn.Module):
    def __init__(self, knowledge_n, exer_n, student_n, usr_emb, itm_emb):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n 
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(NCDNet, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        #semantic rep
        self.usrprf_embeds = torch.tensor(usr_emb).float().cuda()
        self.itmprf_embeds = torch.tensor(itm_emb).float().cuda()
        self.mlp_stu = nn.Sequential(
            nn.Linear(self.usrprf_embeds.shape[1], (self.usrprf_embeds.shape[1] + self.stu_dim) // 2),
            nn.ReLU(),
            nn.Linear((self.usrprf_embeds.shape[1] + self.stu_dim) // 2, self.stu_dim)
        )
        self.mlp_pro = nn.Sequential(
            nn.Linear(self.itmprf_embeds.shape[1], (self.itmprf_embeds.shape[1] + 2000) // 2),
            nn.ReLU(),
            nn.Linear((self.itmprf_embeds.shape[1] + 2000) // 2, 1)
        )
        self._init_weight()

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        # for p in self.parameters():
        #     p.requires_grad = False

    def _init_weight(self):
        for m in self.mlp_pro:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        for m in self.mlp_stu:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))
        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise)) 
        
        
        input_x = input_knowledge_point * (stat_emb - k_difficulty) * e_difficulty
        
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)
    
    def get_embeddings(self):
        return self.student_emb

class NCDNet_con(nn.Module):
    def __init__(self, knowledge_n, exer_n, student_n, usr_emb, itm_emb, similar_students, similar_exercises):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n 
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(NCDNet_con, self).__init__()

        # prediction sub-net
        self.similar_students = torch.tensor([similar_students[key] for key in similar_students]).cuda()
        self.similar_exercises = torch.tensor([similar_exercises[key] for key in similar_exercises]).cuda()
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        #semantic rep
        self.usrprf_embeds = torch.tensor(usr_emb).float().cuda()
        self.itmprf_embeds = torch.tensor(itm_emb).float().cuda()
        self.mlp_stu = nn.Sequential(
            nn.Linear(self.usrprf_embeds.shape[1], (self.usrprf_embeds.shape[1] + self.stu_dim) // 2),
            nn.ReLU(),
            nn.Linear((self.usrprf_embeds.shape[1] + self.stu_dim) // 2, self.stu_dim)
        )
        self.mlp_pro = nn.Sequential(
            nn.Linear(self.itmprf_embeds.shape[1], (self.itmprf_embeds.shape[1] + 2000) // 2),
            nn.ReLU(),
            nn.Linear((self.itmprf_embeds.shape[1] + 2000) // 2, 1+self.knowledge_dim)
        )
        self._init_weight()

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def _init_weight(self):
        for m in self.mlp_pro:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        for m in self.mlp_stu:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))
        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise)) 

        
        input_x = input_knowledge_point * (stat_emb - k_difficulty) * e_difficulty
        
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))


        stu_emb = self.student_emb(stu_id)
        pro_emb = torch.cat((self.e_difficulty(input_exercise), self.k_difficulty(input_exercise)), dim=1)

        
        usrprf_embeds = self.mlp_stu(self.usrprf_embeds)
        itmprf_embeds = self.mlp_pro(self.itmprf_embeds)
        usr_prf_embeds = usrprf_embeds[stu_id]
        itm_prf_embeds = itmprf_embeds[input_exercise]

        simi_user_idx = self.similar_students[stu_id]
        simi_exer_idx = self.similar_exercises[input_exercise]
        simi_user = usrprf_embeds[simi_user_idx].reshape(-1, self.stu_dim)
        simi_exer = itmprf_embeds[simi_exer_idx].reshape(-1, 1+self.knowledge_dim)
        
        
        
        kd_loss = cal_infonce_loss(stu_emb, usr_prf_embeds, usrprf_embeds, 0.5)/stu_emb.shape[0] + cal_infonce_loss(pro_emb, itm_prf_embeds, itmprf_embeds, 0.5)/pro_emb.shape[0]
        simi_loss = cal_infonce_loss(stu_emb, usr_prf_embeds, simi_user, 0.5)/stu_emb.shape[0] + cal_infonce_loss(pro_emb, itm_prf_embeds, simi_exer, 0.5)/pro_emb.shape[0]
        
        total_loss =  0.04*simi_loss + 0.015*kd_loss 
        return output_1.view(-1) ,total_loss

    def get_embeddings(self):
        return self.student_emb
    

class NCDNet_gen(nn.Module):
    def __init__(self, knowledge_n, exer_n, student_n, usr_emb, itm_emb, stu_ratio, exer_ratio):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n 
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable
        self.mode = 0

        super(NCDNet_gen, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        #semantic rep
        self.usrprf_embeds = torch.tensor(usr_emb).float().cuda()
        self.itmprf_embeds = torch.tensor(itm_emb).float().cuda()
        self.stu_ratio = torch.tensor(stu_ratio).float().cuda()
        self.exer_ratio = torch.tensor(exer_ratio).float().cuda()
        
        #gnerative pro
        self.masker_stu = DynamicNodeMask(0.05, self.stu_dim)
        self.masker_pro = DynamicNodeMask(0.05, 1)
        self.mlp_stu = nn.Sequential(
            nn.Linear(self.stu_dim, (self.usrprf_embeds.shape[1] + 20000) // 2),
            nn.ReLU(),
            nn.Linear((self.usrprf_embeds.shape[1] + 20000) // 2, self.usrprf_embeds.shape[1])
        )
        self.mlp_pro = nn.Sequential(
            nn.Linear(1, (self.itmprf_embeds.shape[1] + 10000) // 2),
            nn.ReLU(),
            nn.Linear((self.itmprf_embeds.shape[1] + 10000) // 2, self.itmprf_embeds.shape[1])
        )
        

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        self._init_weight()

    def _init_weight(self):
        for m in self.mlp_stu:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        for m in self.mlp_pro:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    
    def _mask_stu(self,embeds, ratio):

        masked_embeds, seeds = self.masker_stu(embeds, ratio)
        return masked_embeds, seeds

    def _mask_pro(self,embeds, ratio):

        masked_embeds, seeds = self.masker_pro(embeds, ratio)
        return masked_embeds, seeds

    def forward(self, stu_id, input_exercise, input_knowledge_point):

        if self.mode == 0:
            stu_emb = self.student_emb(stu_id)
            exer_emb = self.e_difficulty(input_exercise)

            stu_ratio = self.stu_ratio[stu_id]
            exer_ratio = self.exer_ratio[input_exercise]

            masked_stu_emb, stu_seeds = self._mask_stu(stu_emb, stu_ratio)
            masked_pro_emb, pro_seeds = self._mask_pro(exer_emb, exer_ratio)

            stu_semantic = self.usrprf_embeds[stu_id]
            exer_semantic = self.itmprf_embeds[input_exercise]
            stat_emb = torch.sigmoid(masked_stu_emb)
            e_difficulty = torch.sigmoid(masked_pro_emb)
            k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))

            input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
            input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
            input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
            output_1 = torch.sigmoid(self.prednet_full3(input_x))


            recon_loss = 2 * self._reconstruction(masked_stu_emb,masked_pro_emb,stu_semantic,exer_semantic)

            return output_1.view(-1), recon_loss
        else:
            stu_emb = self.student_emb(stu_id)
            stat_emb = torch.sigmoid(stu_emb)
            k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))
            e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise)) 
            
            
            input_x = input_knowledge_point * (stat_emb - k_difficulty) * e_difficulty
            
            input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
            input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
            output_1 = torch.sigmoid(self.prednet_full3(input_x))

            return output_1.view(-1)

    def get_embeddings(self):
        return self.student_emb
    
    def _reconstruction(self, stu_embeds, pro_embeds, stu_semantic, exer_semantic):
        cdm_stu_embeds = self.mlp_stu(stu_embeds)
        cdm_exer_embeds = self.mlp_pro(pro_embeds)
        
        recon_loss = ssl_con_loss(cdm_stu_embeds, stu_semantic, 0.5) + ssl_con_loss(cdm_exer_embeds, exer_semantic, 0.5)
        return recon_loss

    def cal_loss(self):

        return self.recon_loss






class NCD_gen(CDM):
    '''Neural Cognitive Diagnosis Model'''
    def __init__(self, knowledge_n, exer_n, student_n, log_name, best_model, usr_emb, itm_emb, stu_ratio, exer_ratio):
        super(NCD_gen, self).__init__()
        self.ncdm_net = NCDNet_gen(knowledge_n, exer_n, student_n, usr_emb, itm_emb, stu_ratio, exer_ratio)
        self.log_name = log_name
        self.best_model = best_model

    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.002, silence=False):
        last_time = time.time()
        best_auc = 0
        best_rmse = 100
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.ncdm_net.parameters(), lr=lr)
        for epoch_i in range(epoch):
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                batch_count += 1
                user_id, item_id, knowledge_emb, y = batch_data
                # print(user_id)
                # print(item_id)
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                y: torch.Tensor = (y>0.5).float().to(device)
                pred, recon_loss = self.ncdm_net(user_id, item_id, knowledge_emb)
                bce_loss = loss_function(pred, y)
                loss = bce_loss + recon_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            now = time.time()
            print(f'[Epoch {epoch_i}] average loss on train: {np.mean(epoch_losses):1.6f} time cost: {(now - last_time):1.6f}')
            with open(self.log_name, 'a') as f:
                f.write(f"[Epoch {epoch_i}] average loss on train: {np.mean(epoch_losses):1.6f} time cost: {now - last_time:1.6f}"+"\n")
            last_time = now


            if test_data is not None:
                auc, accuracy, rmse = self.eval(test_data, device=device)
                if auc > best_auc:
                    best_auc = auc
                    print("Update best auc!")
                    with open(self.log_name, 'a') as f:
                        f.write("Update best auc!")
                    self.save(self.best_model)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f, rmse: %.6f" % (epoch_i, auc, accuracy, rmse))
                with open(self.log_name, 'a') as f:
                    f.write("[Epoch %d] auc: %.6f, accuracy: %.6f, rmse: %.6f" % (epoch_i, auc, accuracy, rmse)+"\n")


    def eval(self, test_data, device="cpu", threshold=0.6, save=False):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.eval()
        self.ncdm_net.mode = 1
        y_true, y_pred = [], []
        iid, uid = [], []
        correct_count, exer_count = 0, 0
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, knowledge_emb, y = batch_data
            uid.extend(user_id.tolist())
            iid.extend(item_id.tolist())
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred = self.ncdm_net(user_id, item_id, knowledge_emb)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        for i in range(len(y_true)):
            if (y_true[i] >= 0.5 and y_pred[i] >= 0.5) or (y_true[i] == 0 and y_pred[i] < 0.5):
                correct_count += 1
        exer_count+=len(y_true)
        auc = roc_auc_score(np.array(y_true) >= 0.5, y_pred)
        acc = correct_count / exer_count
        rmse = mean_squared_error(np.array(y_true) >= 0.5, y_pred, squared=False)
        self.ncdm_net.mode = 0
        with open(self.log_name, 'a') as f:
            f.write("Test auc: %.6f, accuracy: %.6f, rmse: %.6f" % (auc, acc, rmse)+"\n")
        return auc, acc, rmse


    def save(self, filepath):
        torch.save(self.ncdm_net.state_dict(), filepath)

    def load(self, filepath):
        self.ncdm_net.load_state_dict(torch.load(filepath))  # , map_location=lambda s, loc: s
    
    def get_embeddings(self):
        return self.ncdm_net.get_embeddings()

class NCD_con(CDM):
    '''Neural Cognitive Diagnosis Model'''
    def __init__(self, knowledge_n, exer_n, student_n, log_name, best_model, usr_emb, itm_emb, similar_students, similar_exercises):
        super(NCD_con, self).__init__()
        
        self.ncdm_net = NCDNet_con(knowledge_n, exer_n, student_n, usr_emb, itm_emb, similar_students, similar_exercises)
        self.log_name = log_name
        self.best_model = best_model

    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.002, silence=False):
        last_time = time.time()
        best_auc = 0
        best_rmse = 100
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.ncdm_net.parameters(), lr=lr)
        for epoch_i in range(epoch):
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                batch_count += 1
                user_id, item_id, knowledge_emb, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                y: torch.Tensor = (y>0.5).float().to(device)
                pred, recon_loss = self.ncdm_net(user_id, item_id, knowledge_emb)


                bce_loss = loss_function(pred, y)
                loss = bce_loss + recon_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            now = time.time()
            print(f'[Epoch {epoch_i}] average loss on train: {np.mean(epoch_losses):1.6f} time cost: {(now - last_time):1.6f}')
            with open(self.log_name, 'a') as f:
                f.write(f"[Epoch {epoch_i}] average loss on train: {np.mean(epoch_losses):1.6f} time cost: {now - last_time:1.6f}"+"\n")
            last_time = now


            if test_data is not None:
                auc, accuracy, rmse = self.eval(test_data, device=device)
                if auc > best_auc:
                    best_auc = auc
                    print("Update best auc!")
                    with open(self.log_name, 'a') as f:
                        f.write("Update best auc!")
                    self.save(self.best_model)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f, rmse: %.6f" % (epoch_i, auc, accuracy, rmse))
                with open(self.log_name, 'a') as f:
                    f.write("[Epoch %d] auc: %.6f, accuracy: %.6f, rmse: %.6f" % (epoch_i, auc, accuracy, rmse)+"\n")


    def eval(self, test_data, device="cpu", threshold=0.6, save=False):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.eval()
        y_true, y_pred = [], []
        uid, iid = [], []
        correct_count, exer_count = 0, 0
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, knowledge_emb, y = batch_data
            uid.extend(user_id.tolist())
            iid.extend(item_id.tolist())
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred,_ = self.ncdm_net(user_id, item_id, knowledge_emb)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        for i in range(len(y_true)):
            if (y_true[i] >= 0.5 and y_pred[i] >= 0.5) or (y_true[i] == 0 and y_pred[i] < 0.5):
                correct_count += 1
        exer_count+=len(y_true)
        auc = roc_auc_score(np.array(y_true) >= 0.5, y_pred)
        acc = correct_count / exer_count
        rmse = mean_squared_error(np.array(y_true) >= 0.5, y_pred, squared=False)
        with open(self.log_name, 'a') as f:
            f.write("Test auc: %.6f, accuracy: %.6f, rmse: %.6f" % (auc, acc, rmse)+"\n")
        return auc, acc, rmse


    def save(self, filepath):
        torch.save(self.ncdm_net.state_dict(), filepath)

    def load(self, filepath):
        self.ncdm_net.load_state_dict(torch.load(filepath)) 

    def get_embeddings(self):
        return self.ncdm_net.get_embeddings()



class NCD(CDM):
    '''Neural Cognitive Diagnosis Model'''
    def __init__(self, knowledge_n, exer_n, student_n, log_name, best_model, usr_emb, itm_emb):
        super(NCD, self).__init__()
        self.ncdm_net = NCDNet(knowledge_n, exer_n, student_n, usr_emb, itm_emb)
        self.log_name = log_name
        self.best_model = best_model

    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.002, silence=False):
        last_time = time.time()
        best_auc = 0
        best_rmse = 100
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.ncdm_net.parameters(), lr=lr)
        for epoch_i in range(epoch):
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                batch_count += 1
                user_id, item_id, knowledge_emb, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                y: torch.Tensor = (y>0.5).float().to(device)
                pred = self.ncdm_net(user_id, item_id, knowledge_emb)

                bce_loss = loss_function(pred, y)
                loss = bce_loss 

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            now = time.time()
            print(f'[Epoch {epoch_i}] average loss on train: {np.mean(epoch_losses):1.6f} time cost: {(now - last_time):1.6f}')
            with open(self.log_name, 'a') as f:
                f.write(f"[Epoch {epoch_i}] average loss on train: {np.mean(epoch_losses):1.6f} time cost: {now - last_time:1.6f}"+"\n")
            last_time = now


            if test_data is not None:
                auc, accuracy, rmse = self.eval(test_data, device=device)
                if auc > best_auc:
                    best_auc = auc
                    print("Update best auc!")
                    with open(self.log_name, 'a') as f:
                        f.write("Update best auc!")
                    self.save(self.best_model)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f, rmse: %.6f" % (epoch_i, auc, accuracy, rmse))
                with open(self.log_name, 'a') as f:
                    f.write("[Epoch %d] auc: %.6f, accuracy: %.6f, rmse: %.6f" % (epoch_i, auc, accuracy, rmse)+"\n")


    def eval(self, test_data, device="cpu", threshold=0.6, save=False):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.eval()
        y_true, y_pred = [], []
        uid, iid = [], []
        correct_count, exer_count = 0, 0
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, knowledge_emb, y = batch_data
            uid.extend(user_id.tolist())
            iid.extend(item_id.tolist())
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred = self.ncdm_net(user_id, item_id, knowledge_emb)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())


        for i in range(len(y_true)):
            if (y_true[i] >= 0.5 and y_pred[i] >= 0.5) or (y_true[i] == 0 and y_pred[i] < 0.5):
                correct_count += 1
        exer_count+=len(y_true)
        auc = roc_auc_score(np.array(y_true) >= 0.5, y_pred)
        acc = correct_count / exer_count
        rmse = mean_squared_error(np.array(y_true) >= 0.5, y_pred, squared=False)
        with open(self.log_name, 'a') as f:
            f.write("Test auc: %.6f, accuracy: %.6f, rmse: %.6f" % (auc, acc, rmse)+"\n")
        return auc, acc, rmse


    def save(self, filepath):
        torch.save(self.ncdm_net.state_dict(), filepath)

    def load(self, filepath):
        self.ncdm_net.load_state_dict(torch.load(filepath))  # , map_location=lambda s, loc: s
    
    def get_embeddings(self):
        return self.ncdm_net.get_embeddings()