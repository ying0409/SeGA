import pytorch_lightning as pl
from torch import nn
from layer import RGTLayer
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score, roc_auc_score, mean_squared_error
import globals
import numpy as np
import math
import torch_geometric
import random
import os
import sys
import torch.nn.functional as F

class SeGA(pl.LightningModule):
    def __init__(self, args, pretrain):
        super(SeGA, self).__init__()
        self.device_name = args.device

        self.pretrain_lr = args.pretrain_lr
        self.finetune_lr = args.finetune_lr
        self.l2_reg = args.l2_reg
        self.pretext_task = args.pretext_task
        self.pretrain = pretrain
        if pretrain == True:
            prompt_embedding_path = "/home/yychang/Sega/datasets/processed_data/sample/prompt_embeddings/"
            self.pretrain_embedding_dict = torch.load(prompt_embedding_path + "{}_{}.pt".format(args.prompt_encoder, args.template), map_location='cpu')
            if args.template in ["l", "s", "n"]:
                self.none_zero_dict = np.array(torch.load(prompt_embedding_path + "none_zero.pt", map_location='cpu'))
            else:
                self.none_zero_dict = np.array(torch.load(prompt_embedding_path + "none_zero_{}.pt".format(args.template), map_location='cpu'))
        if args.pretext_task == "multi":
            self.multi_classifier = nn.Linear(args.user_channel, 153)
            self.BCELoss = nn.BCEWithLogitsLoss()
        self.neg_num = args.neg_num
        self.T = args.T

        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.linear_channels = args.linear_channels
        
        self.user_num = args.user_num
        self.user_numeric_num = args.user_numeric_num
        self.user_cat_num = args.user_cat_num
        self.user_des_channel = args.user_des_channel
        self.user_tweet_channel = args.user_tweet_channel
        
        self.list = args.lst
        self.list_num = args.list_num
        self.list_numeric_num = args.list_numeric_num
        self.list_cat_num = args.list_cat_num
        self.list_des_channel = args.list_des_channel
        self.list_tweet_channel = args.list_tweet_channel
        
        self.user_in_linear_numeric = nn.Linear(args.user_numeric_num, int(args.linear_channels/4), bias=True)
        self.user_in_linear_bool = nn.Linear(args.user_cat_num, int(args.linear_channels/4), bias=True)
        self.user_in_linear_tweet = nn.Linear(args.user_tweet_channel, int(args.linear_channels/4), bias=True)
        self.user_in_linear_des = nn.Linear(args.user_des_channel, int(args.linear_channels/4), bias=True)
        self.user_linear = nn.Linear(args.linear_channels, args.linear_channels)
        
        if args.lst:
            self.list_in_linear_numeric = nn.Linear(args.list_numeric_num, int(args.linear_channels/4), bias=True)
            self.list_in_linear_bool = nn.Linear(args.list_cat_num, int(args.linear_channels/4), bias=True)
            self.list_in_linear_tweet = nn.Linear(args.list_tweet_channel, int(args.linear_channels/4), bias=True)
            self.list_in_linear_des = nn.Linear(args.list_des_channel, int(args.linear_channels/4), bias=True)
            self.list_linear = nn.Linear(args.linear_channels, args.linear_channels)

        self.RGT_layer1 = RGTLayer(num_edge_type=args.num_edge_type, in_channels=args.linear_channels, out_channels=args.out_channel, trans_heads=args.trans_head, semantic_head=args.semantic_head, dropout=args.dropout)
        self.RGT_layer2 = RGTLayer(num_edge_type=args.num_edge_type, in_channels=args.linear_channels, out_channels=args.out_channel, trans_heads=args.trans_head, semantic_head=args.semantic_head, dropout=args.dropout)

        self.out = nn.Linear(args.out_channel,args.user_channel)
        self.prompt_mlp = nn.Linear(768,args.user_channel)
        self.user_mlp = nn.Linear(args.user_channel,args.user_channel)
        self.classifier = nn.Linear(args.user_channel, 3)

        self.drop = nn.Dropout(args.dropout)
        self.CELoss = nn.CrossEntropyLoss()
        self.ReLU = nn.LeakyReLU()
        
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def training_step(self, train_batch, batch_idx):
        ### Node Feature Encoding ###
        # create an empty tensor to store the features of all nodes
        features = torch.empty(train_batch.size(0),self.linear_channels,dtype=torch.float16,device=self.device_name)
        
        # obtain the features of user nodes
        user_mask = torch.Tensor((train_batch.n_id >= 0) & (train_batch.n_id < 100001))
        user_x = train_batch.x.clone()
        user_x = user_x[user_mask]
        user_cat_features = user_x[:, :self.user_cat_num]
        user_prop_features = user_x[:, self.user_cat_num: self.user_cat_num + self.user_numeric_num]
        user_tweet_features = user_x[:, self.user_cat_num+self.user_numeric_num: self.user_cat_num+self.user_numeric_num+self.user_tweet_channel]
        user_des_features = user_x[:, self.user_cat_num+self.user_numeric_num+self.user_tweet_channel: self.user_cat_num+self.user_numeric_num+self.user_tweet_channel+self.user_des_channel]
        
        user_features_numeric = self.drop(self.ReLU(self.user_in_linear_numeric(user_prop_features)))
        user_features_bool = self.drop(self.ReLU(self.user_in_linear_bool(user_cat_features)))
        user_features_tweet = self.drop(self.ReLU(self.user_in_linear_tweet(user_tweet_features)))
        user_features_des = self.drop(self.ReLU(self.user_in_linear_des(user_des_features)))

        user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim=1)
        user_features = self.drop(self.ReLU(self.user_linear(user_features)))
        features[user_mask] = user_features

        # obtain the features of list nodes
        if self.list:
            list_mask = torch.Tensor((train_batch.n_id >= 100001) & (train_batch.n_id < 120789))
            list_x = train_batch.x.clone()
            list_x = list_x[list_mask]
            list_cat_features = list_x[:, :self.list_cat_num]
            list_prop_features = list_x[:, self.list_cat_num: self.list_cat_num + self.list_numeric_num]
            list_tweet_features = list_x[:, self.list_cat_num+self.list_numeric_num: self.list_cat_num+self.list_numeric_num+self.list_tweet_channel]
            list_des_features = list_x[:, self.list_cat_num+self.list_numeric_num+self.list_tweet_channel: self.list_cat_num+self.list_numeric_num+self.list_tweet_channel+self.list_des_channel]

            list_features_numeric = self.drop(self.ReLU(self.list_in_linear_numeric(list_prop_features)))
            list_features_bool = self.drop(self.ReLU(self.list_in_linear_bool(list_cat_features)))
            list_features_tweet = self.drop(self.ReLU(self.list_in_linear_tweet(list_tweet_features)))
            list_features_des = self.drop(self.ReLU(self.list_in_linear_des(list_des_features)))

            list_features = torch.cat((list_features_numeric,list_features_bool,list_features_tweet,list_features_des), dim = 1)
            list_features = self.drop(self.ReLU(self.list_linear(list_features)))
            features[list_mask] = list_features

        ### Heterogeneous Encoder ###
        # input the features into the RGT layer
        edge_index = train_batch.edge_index
        edge_type = train_batch.edge_attr.view(-1)
        features = self.ReLU(self.RGT_layer1(features, edge_index, edge_type))
        features = self.ReLU(self.RGT_layer2(features, edge_index, edge_type))
        
        if self.pretrain == True:
            ### Pretraining ###
            # select the features of user nodes which are in the train_idx
            indices = train_batch.input_id
            batch_user = train_batch.pretrain_train_idx[indices]
            mask = np.isin(train_batch.n_id.cpu(), batch_user.cpu()).astype(int)
            mask = torch.BoolTensor(mask)

            label = train_batch.y
            label = label[mask]
            if self.pretext_task == "contrastive":
                ### Transform the User Embedding with Contrastive Classifier ###
                features = self.drop(self.ReLU(self.out(features)))
                anchor = self.user_mlp(features)
                anchor = anchor[mask.nonzero(as_tuple=True)[0]]

                ### Transform the Pseudo-Label Embedding with Contrastive Classifier ###
                trans_pretrain_embedding_dict = self.prompt_mlp(self.pretrain_embedding_dict.to(self.device_name))
                pos_embedding = trans_pretrain_embedding_dict[torch.tensor(label)]

                # Negative Pairs Sampling
                neg_index = []
                all_samples = self.none_zero_dict
                for i in range(anchor.size(0)):
                    neg_index.append(random.choices(all_samples[all_samples != label[i].item()], k=self.neg_num))            
                neg_embedding = trans_pretrain_embedding_dict[torch.tensor(neg_index)]

                split_tensors = torch.split(neg_embedding, 1, dim=1)

                ### Contrastive Loss ###
                l_neg = []
                l_pos = F.cosine_similarity(anchor, pos_embedding).unsqueeze(-1)
                for i in range(self.neg_num):
                    l_neg.append(F.cosine_similarity(anchor, torch.reshape(split_tensors[i], (anchor.size(0), -1))).unsqueeze(-1))

                logits = torch.cat([l_pos, torch.cat(l_neg,dim=1)], dim=1)
                logits /= self.T

                label = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

                loss = self.CELoss(logits.float(), label)
                self.log("pre_train_ce", loss, prog_bar=True)
                return loss

            elif self.pretext_task == "multi":
                features = self.drop(self.ReLU(self.out(features)))
                pred = self.multi_classifier(features)
                pred = pred[mask.nonzero(as_tuple=True)[0]]

                loss = self.BCELoss(pred.float(),label.float())
                return loss
        else:
            ### Finetuning ###
            # mask the result of list nodes (select user node than it's n_id is in train_idx)
            indices = train_batch.input_id
            batch_user = train_batch.finetune_train_idx[indices]
            mask = np.isin(train_batch.n_id.cpu(), batch_user.cpu()).astype(int)
            mask = torch.BoolTensor(mask)

            label = train_batch.y
            label = label[mask]

            ### Detection Classifier ###
            features = self.drop(self.ReLU(self.out(features)))
            pred = self.classifier(features)
            pred = pred[mask.nonzero(as_tuple=True)[0]]

            pred_binary = torch.argmax(pred, dim=1)
            f1 = f1_score(label.cpu(), pred_binary.cpu(),average="macro")
            self.log("train_f1", f1, prog_bar=True)

            loss = self.CELoss(pred.float(), label.long())

            return loss

    def validation_step(self, valid_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            features = torch.empty(valid_batch.size(0),self.linear_channels,dtype=torch.float16,device=self.device_name)
            
            user_mask = torch.Tensor((valid_batch.n_id >= 0) & (valid_batch.n_id < 100001))
            user_x = valid_batch.x.clone()
            user_x = user_x[user_mask]
            user_cat_features = user_x[:, :self.user_cat_num]
            user_prop_features = user_x[:, self.user_cat_num: self.user_cat_num + self.user_numeric_num]
            user_tweet_features = user_x[:, self.user_cat_num+self.user_numeric_num: self.user_cat_num+self.user_numeric_num+self.user_tweet_channel]
            user_des_features = user_x[:, self.user_cat_num+self.user_numeric_num+self.user_tweet_channel: self.user_cat_num+self.user_numeric_num+self.user_tweet_channel+self.user_des_channel]

            user_features_numeric = self.drop(self.ReLU(self.user_in_linear_numeric(user_prop_features)))
            user_features_bool = self.drop(self.ReLU(self.user_in_linear_bool(user_cat_features)))
            user_features_tweet = self.drop(self.ReLU(self.user_in_linear_tweet(user_tweet_features)))
            user_features_des = self.drop(self.ReLU(self.user_in_linear_des(user_des_features)))
            
            user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)
            user_features = self.drop(self.ReLU(self.user_linear(user_features)))
            features[user_mask] = user_features
            
            if self.list:
                list_mask = torch.Tensor((valid_batch.n_id >= 100001) & (valid_batch.n_id < 120789))
                list_x = valid_batch.x.clone()
                list_x = list_x[list_mask]
                list_cat_features = list_x[:, :self.list_cat_num]
                list_prop_features = list_x[:, self.list_cat_num: self.list_cat_num + self.list_numeric_num]
                list_tweet_features = list_x[:, self.list_cat_num+self.list_numeric_num: self.list_cat_num+self.list_numeric_num+self.list_tweet_channel]
                list_des_features = list_x[:, self.list_cat_num+self.list_numeric_num+self.list_tweet_channel: self.list_cat_num+self.list_numeric_num+self.list_tweet_channel+self.list_des_channel]
            
                list_features_numeric = self.drop(self.ReLU(self.list_in_linear_numeric(list_prop_features)))
                list_features_bool = self.drop(self.ReLU(self.list_in_linear_bool(list_cat_features)))
                list_features_tweet = self.drop(self.ReLU(self.list_in_linear_tweet(list_tweet_features)))
                list_features_des = self.drop(self.ReLU(self.list_in_linear_des(list_des_features)))
                
                list_features = torch.cat((list_features_numeric,list_features_bool,list_features_tweet,list_features_des), dim = 1)
                list_features = self.drop(self.ReLU(self.list_linear(list_features)))
                features[list_mask] = list_features
            
            edge_index = valid_batch.edge_index
            edge_type = valid_batch.edge_attr.view(-1)
            features = self.ReLU(self.RGT_layer1(features, edge_index, edge_type))
            features = self.ReLU(self.RGT_layer2(features, edge_index, edge_type))

            features = self.drop(self.ReLU(self.out(features)))
            pred = self.classifier(features)
            
            # mask the result of list nodes (select user node than it's n_id is in train_idx)
            mask = np.isin(valid_batch.n_id.cpu(), valid_batch.finetune_valid_idx.cpu()[self.batch_size*batch_idx:self.batch_size*(batch_idx+1)]).astype(int)
            mask = torch.BoolTensor(mask)
            pred = pred[mask.nonzero(as_tuple=True)[0]]
            pred_binary = torch.argmax(pred, dim=1)
            label = valid_batch.y
            label = label[mask]

            globals.fine_pred_val += list(pred_binary.squeeze().cpu())
            globals.fine_label_val += list(label.squeeze().cpu())
    
    def validation_epoch_end(self, outputs):
        precision = precision_score(globals.fine_label_val, globals.fine_pred_val,average="macro",zero_division=0)
        recall = recall_score(globals.fine_label_val, globals.fine_pred_val,average="macro",zero_division=0)
        f1 = f1_score(globals.fine_label_val, globals.fine_pred_val,average="macro",zero_division=0)

        globals.fine_label_val = []
        globals.fine_pred_val = []

        self.log('val_f1', f1, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            features = torch.empty(test_batch.size(0),self.linear_channels,dtype=torch.float16,device=self.device_name)
            
            user_mask = torch.Tensor((test_batch.n_id >= 0) & (test_batch.n_id < 100001))
            user_x = test_batch.x.clone()
            user_x = user_x[user_mask]
            user_cat_features = user_x[:, :self.user_cat_num]
            user_prop_features = user_x[:, self.user_cat_num: self.user_cat_num + self.user_numeric_num]
            user_tweet_features = user_x[:, self.user_cat_num+self.user_numeric_num: self.user_cat_num+self.user_numeric_num+self.user_tweet_channel]
            user_des_features = user_x[:, self.user_cat_num+self.user_numeric_num+self.user_tweet_channel: self.user_cat_num+self.user_numeric_num+self.user_tweet_channel+self.user_des_channel]

            user_features_numeric = self.drop(self.ReLU(self.user_in_linear_numeric(user_prop_features)))
            user_features_bool = self.drop(self.ReLU(self.user_in_linear_bool(user_cat_features)))
            user_features_tweet = self.drop(self.ReLU(self.user_in_linear_tweet(user_tweet_features)))
            user_features_des = self.drop(self.ReLU(self.user_in_linear_des(user_des_features)))
            
            user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)
            user_features = self.drop(self.ReLU(self.user_linear(user_features)))
            features[user_mask] = user_features

            if self.list:
                list_mask = torch.Tensor((test_batch.n_id >= 100001) & (test_batch.n_id < 120789))
                list_x = test_batch.x.clone()
                list_x = list_x[list_mask]
                list_cat_features = list_x[:, :self.list_cat_num]
                list_prop_features = list_x[:, self.list_cat_num: self.list_cat_num + self.list_numeric_num]
                list_tweet_features = list_x[:, self.list_cat_num+self.list_numeric_num: self.list_cat_num+self.list_numeric_num+self.list_tweet_channel]
                list_des_features = list_x[:, self.list_cat_num+self.list_numeric_num+self.list_tweet_channel: self.list_cat_num+self.list_numeric_num+self.list_tweet_channel+self.list_des_channel]
            
                list_features_numeric = self.drop(self.ReLU(self.list_in_linear_numeric(list_prop_features)))
                list_features_bool = self.drop(self.ReLU(self.list_in_linear_bool(list_cat_features)))
                list_features_tweet = self.drop(self.ReLU(self.list_in_linear_tweet(list_tweet_features)))
                list_features_des = self.drop(self.ReLU(self.list_in_linear_des(list_des_features)))
                
                list_features = torch.cat((list_features_numeric,list_features_bool,list_features_tweet,list_features_des), dim = 1)
                list_features = self.drop(self.ReLU(self.list_linear(list_features)))
                features[list_mask] = list_features
            
            edge_index = test_batch.edge_index
            edge_type = test_batch.edge_attr.view(-1)            
            features = self.ReLU(self.RGT_layer1(features, edge_index, edge_type))
            features = self.ReLU(self.RGT_layer2(features, edge_index, edge_type))

            features = self.drop(self.ReLU(self.out(features)))
            pred = self.classifier(features)
            
            # mask the result of list nodes (select user node than it's n_id is in test_idx)
            mask = np.isin(test_batch.n_id.cpu(), test_batch.finetune_test_idx.cpu()[self.test_batch_size*batch_idx:self.test_batch_size*(batch_idx+1)]).astype(int)
            mask = torch.BoolTensor(mask)
            pred = pred[mask.nonzero(as_tuple=True)[0]]
            pred_binary = torch.argmax(pred, dim=1)
            label = test_batch.y
            label = label[mask]

            user_embedding = features[mask.nonzero(as_tuple=True)[0]]
            globals.all_fine_user_embedding += user_embedding.squeeze().cpu().numpy().tolist()
            globals.fine_pred_test += list(pred_binary.squeeze().cpu())
            globals.fine_pred_test_prob += pred.squeeze().cpu().numpy().tolist()
            globals.fine_label_test += list(label.squeeze().cpu())

            acc = accuracy_score(label.cpu(), pred_binary.cpu())
            f1 = f1_score(label.cpu(), pred_binary.cpu(),average="macro")
            precision =precision_score(label.cpu(), pred_binary.cpu(),average="macro")
            recall = recall_score(label.cpu(), pred_binary.cpu(),average="macro")
            bacc = balanced_accuracy_score(label.cpu(), pred_binary.cpu())

            self.log("acc", acc)
            self.log("f1",f1)
            self.log("precision", precision)
            self.log("recall", recall)
            self.log("bacc",bacc)

    def configure_optimizers(self):
        if self.pretrain == True:
            lr = self.pretrain_lr
        else:
            lr = self.finetune_lr
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=self.l2_reg, amsgrad=False)
        scheduler = CosineAnnealingLR(optimizer, T_max=16, eta_min=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }