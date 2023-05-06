import pytorch_lightning as pl
from torch import nn
from layer import RGTLayer
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score, roc_auc_score, mean_squared_error
import globals
from focal_loss import FocalLoss

class RGTPretrain(pl.LightningModule):
    def __init__(self, args, pretrain):
        super(RGTPretrain, self).__init__()

        self.args = args
        self.lr = args.lr
        self.l2_reg = args.l2_reg
        self.pretrain = pretrain
    
        self.user_in_linear_numeric = nn.Linear(args.user_numeric_num, int(args.linear_channels/4), bias=True)
        self.user_in_linear_bool = nn.Linear(args.user_cat_num, int(args.linear_channels/4), bias=True)
        self.user_in_linear_tweet = nn.Linear(args.user_tweet_channel, int(args.linear_channels/4), bias=True)
        self.user_in_linear_des = nn.Linear(args.user_des_channel, int(args.linear_channels/4), bias=True)
        self.user_linear = nn.Linear(args.linear_channels, args.linear_channels)
        
        self.list_in_linear_numeric = nn.Linear(args.list_numeric_num, int(args.linear_channels/4), bias=True)
        self.list_in_linear_bool = nn.Linear(args.list_cat_num, int(args.linear_channels/4), bias=True)
        self.list_in_linear_tweet = nn.Linear(args.list_tweet_channel, int(args.linear_channels/4), bias=True)
        self.list_in_linear_des = nn.Linear(args.list_des_channel, int(args.linear_channels/4), bias=True)
        self.list_linear = nn.Linear(args.linear_channels, args.linear_channels)

        self.RGT_layer1 = RGTLayer(num_edge_type=5, in_channels=args.linear_channels, out_channels=args.out_channel, trans_heads=args.trans_head, semantic_head=args.semantic_head, dropout=args.dropout)
        self.RGT_layer2 = RGTLayer(num_edge_type=5, in_channels=args.linear_channels, out_channels=args.out_channel, trans_heads=args.trans_head, semantic_head=args.semantic_head, dropout=args.dropout)

        self.out = torch.nn.Linear(args.out_channel,64)
        self.decoder = torch.nn.Linear(64, 1)
        self.classifier = torch.nn.Linear(64, 3)

        self.drop = nn.Dropout(args.dropout)
        self.CELoss = nn.CrossEntropyLoss()
        self.FCLoss = FocalLoss(alpha=None, gamma=3)
        self.MSELoss = nn.MSELoss()
        self.ReLU = nn.LeakyReLU()
        
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def training_step(self, train_batch, batch_idx):
        user_cat_features = train_batch.x[:self.args.user_num, :self.args.user_cat_num]
        user_prop_features = train_batch.x[:self.args.user_num, self.args.user_cat_num: self.args.user_cat_num + self.args.user_numeric_num]
        user_tweet_features = train_batch.x[:self.args.user_num, self.args.user_cat_num+self.args.user_numeric_num: self.args.user_cat_num+self.args.user_numeric_num+self.args.user_tweet_channel]
        user_des_features = train_batch.x[:self.args.user_num, self.args.user_cat_num+self.args.user_numeric_num+self.args.user_tweet_channel: self.args.user_cat_num+self.args.user_numeric_num+self.args.user_tweet_channel+self.args.user_des_channel]

        list_cat_features = train_batch.x[self.args.user_num: self.args.user_num + self.args.list_num, :self.args.list_cat_num]
        list_prop_features = train_batch.x[self.args.user_num: self.args.user_num + self.args.list_num, self.args.list_cat_num: self.args.list_cat_num + self.args.list_numeric_num]
        list_tweet_features = train_batch.x[self.args.user_num: self.args.user_num + self.args.list_num, self.args.list_cat_num+self.args.list_numeric_num: self.args.list_cat_num+self.args.list_numeric_num+self.args.list_tweet_channel]
        list_des_features = train_batch.x[self.args.user_num: self.args.user_num + self.args.list_num, self.args.list_cat_num+self.args.list_numeric_num+self.args.list_tweet_channel: self.args.list_cat_num+self.args.list_numeric_num+self.args.list_tweet_channel+self.args.list_des_channel]
        
        edge_index = train_batch.edge_index
        edge_type = train_batch.edge_attr.view(-1)
        
        user_features_numeric = self.drop(self.ReLU(self.user_in_linear_numeric(user_prop_features)))
        user_features_bool = self.drop(self.ReLU(self.user_in_linear_bool(user_cat_features)))
        user_features_tweet = self.drop(self.ReLU(self.user_in_linear_tweet(user_tweet_features)))
        user_features_des = self.drop(self.ReLU(self.user_in_linear_des(user_des_features)))
        user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)
        user_features = self.drop(self.ReLU(self.user_linear(user_features)))
        #print('Size of user feature:', user_features.size())
        
        list_features_numeric = self.drop(self.ReLU(self.list_in_linear_numeric(list_prop_features)))
        list_features_bool = self.drop(self.ReLU(self.list_in_linear_bool(list_cat_features)))
        list_features_tweet = self.drop(self.ReLU(self.list_in_linear_tweet(list_tweet_features)))
        list_features_des = self.drop(self.ReLU(self.list_in_linear_des(list_des_features)))
        list_features = torch.cat((list_features_numeric,list_features_bool,list_features_tweet,list_features_des), dim = 1)
        list_features = self.drop(self.ReLU(self.list_linear(list_features)))
        #print('Size of list feature:', list_features.size())
        
        features = torch.cat((user_features,list_features),dim=0)
        #print('Size of final feature:', features.size())

        features = self.ReLU(self.RGT_layer1(features, edge_index, edge_type))
        features = self.ReLU(self.RGT_layer2(features, edge_index, edge_type))
        
        if self.pretrain == True:
            mask = [1 if i in train_batch.pretrain_train_idx else 0 for i in train_batch.n_id]
            mask = torch.BoolTensor(mask)
            features = self.drop(self.ReLU(self.out(features)))
            pred = self.decoder(features)
            pred = pred[torch.reshape(mask,(mask.size(0),1))]
            label = train_batch.y
            label = label[mask]
            loss = self.MSELoss(pred, label.float())
        else:
            mask = [1 if i in train_batch.finetune_train_idx else 0 for i in train_batch.n_id]
            mask = torch.BoolTensor(mask)
            features = self.drop(self.ReLU(self.out(features)))
            pred = self.classifier(features)
            pred = pred[mask.nonzero(as_tuple=True)[0]]
            label = train_batch.y
            label = label[mask]
            loss = self.FCLoss(pred.float(), label.long())

        return loss
    
    def validation_step(self, val_batch, batch_idx):
        print('valid_step')
        self.eval()
        with torch.no_grad():
            user_cat_features = val_batch.x[:self.args.user_num, :self.args.user_cat_num]
            user_prop_features = val_batch.x[:self.args.user_num, self.args.user_cat_num: self.args.user_cat_num + self.args.user_numeric_num]
            user_tweet_features = val_batch.x[:self.args.user_num, self.args.user_cat_num+self.args.user_numeric_num: self.args.user_cat_num+self.args.user_numeric_num+self.args.user_tweet_channel]
            user_des_features = val_batch.x[:self.args.user_num, self.args.user_cat_num+self.args.user_numeric_num+self.args.user_tweet_channel: self.args.user_cat_num+self.args.user_numeric_num+self.args.user_tweet_channel+self.args.user_des_channel]
            
            list_cat_features = val_batch.x[:self.args.list_num, :self.args.list_cat_num]
            list_prop_features = val_batch.x[:self.args.list_num, self.args.list_cat_num: self.args.list_cat_num + self.args.list_numeric_num]
            list_tweet_features = val_batch.x[:self.args.list_num, self.args.list_cat_num+self.args.list_numeric_num: self.args.list_cat_num+self.args.list_numeric_num+self.args.list_tweet_channel]
            list_des_features = val_batch.x[:self.args.list_num, self.args.list_cat_num+self.args.list_numeric_num+self.args.list_tweet_channel: self.args.list_cat_num+self.args.list_numeric_num+self.args.list_tweet_channel+self.args.list_des_channel]
        
            edge_index = val_batch.edge_index
            edge_type = val_batch.edge_attr.view(-1)
            
            user_features_numeric = self.drop(self.ReLU(self.user_in_linear_numeric(user_prop_features)))
            user_features_bool = self.drop(self.ReLU(self.user_in_linear_bool(user_cat_features)))
            user_features_tweet = self.drop(self.ReLU(self.user_in_linear_tweet(user_tweet_features)))
            user_features_des = self.drop(self.ReLU(self.user_in_linear_des(user_des_features)))
            user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)
            user_features = self.drop(self.ReLU(self.user_linear(user_features)))
            #print('Size of user feature:', user_features.size())
            
            list_features_numeric = self.drop(self.ReLU(self.list_in_linear_numeric(list_prop_features)))
            list_features_bool = self.drop(self.ReLU(self.list_in_linear_bool(list_cat_features)))
            list_features_tweet = self.drop(self.ReLU(self.list_in_linear_tweet(list_tweet_features)))
            list_features_des = self.drop(self.ReLU(self.list_in_linear_des(list_des_features)))
            list_features = torch.cat((list_features_numeric,list_features_bool,list_features_tweet,list_features_des), dim = 1)
            list_features = self.drop(self.ReLU(self.list_linear(list_features)))
            #print('Size of list feature:', list_features.size())
            
            features = torch.cat((user_features,list_features),dim=0)
            #print('Size of final feature:', features.size())

            features = self.ReLU(self.RGT_layer1(features, edge_index, edge_type))
            features = self.ReLU(self.RGT_layer2(features, edge_index, edge_type))

            features = self.drop(self.ReLU(self.out1(features)))
            pred = self.out2(features)
            # print(pred.size())
            
            pred_binary = torch.argmax(pred, dim=1)
            pred_binary = pred_binary[:user_features.size(0)]
            # print(self.label[val_batch].size())
            label = val_batch.y
            #print(pred_binary.size())
            #print(label.size())

            acc = accuracy_score(label.cpu(), pred_binary.cpu())
            f1 = f1_score(label.cpu(), pred_binary.cpu())
            
            self.log("val_acc", acc, prog_bar=True)
            self.log("val_f1", f1, prog_bar=True)

            # print("acc: {} f1: {}".format(acc, f1))
    
    def test_step(self, test_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            user_cat_features = test_batch.x[:self.args.user_num, :self.args.user_cat_num]
            user_prop_features = test_batch.x[:self.args.user_num, self.args.user_cat_num: self.args.user_cat_num + self.args.user_numeric_num]
            user_tweet_features = test_batch.x[:self.args.user_num, self.args.user_cat_num+self.args.user_numeric_num: self.args.user_cat_num+self.args.user_numeric_num+self.args.user_tweet_channel]
            user_des_features = test_batch.x[:self.args.user_num, self.args.user_cat_num+self.args.user_numeric_num+self.args.user_tweet_channel: self.args.user_cat_num+self.args.user_numeric_num+self.args.user_tweet_channel+self.args.user_des_channel]

            list_cat_features = test_batch.x[self.args.user_num: self.args.user_num + self.args.list_num, :self.args.list_cat_num]
            list_prop_features = test_batch.x[self.args.user_num: self.args.user_num + self.args.list_num, self.args.list_cat_num: self.args.list_cat_num + self.args.list_numeric_num]
            list_tweet_features = test_batch.x[self.args.user_num: self.args.user_num + self.args.list_num, self.args.list_cat_num+self.args.list_numeric_num: self.args.list_cat_num+self.args.list_numeric_num+self.args.list_tweet_channel]
            list_des_features = test_batch.x[self.args.user_num: self.args.user_num + self.args.list_num, self.args.list_cat_num+self.args.list_numeric_num+self.args.list_tweet_channel: self.args.list_cat_num+self.args.list_numeric_num+self.args.list_tweet_channel+self.args.list_des_channel]
            
            edge_index = test_batch.edge_index
            edge_type = test_batch.edge_attr.view(-1)
            
            user_features_numeric = self.drop(self.ReLU(self.user_in_linear_numeric(user_prop_features)))
            user_features_bool = self.drop(self.ReLU(self.user_in_linear_bool(user_cat_features)))
            user_features_tweet = self.drop(self.ReLU(self.user_in_linear_tweet(user_tweet_features)))
            user_features_des = self.drop(self.ReLU(self.user_in_linear_des(user_des_features)))
            user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)
            user_features = self.drop(self.ReLU(self.user_linear(user_features)))
            #print('Size of user feature:', user_features.size())
            
            list_features_numeric = self.drop(self.ReLU(self.list_in_linear_numeric(list_prop_features)))
            list_features_bool = self.drop(self.ReLU(self.list_in_linear_bool(list_cat_features)))
            list_features_tweet = self.drop(self.ReLU(self.list_in_linear_tweet(list_tweet_features)))
            list_features_des = self.drop(self.ReLU(self.list_in_linear_des(list_des_features)))
            list_features = torch.cat((list_features_numeric,list_features_bool,list_features_tweet,list_features_des), dim = 1)
            list_features = self.drop(self.ReLU(self.list_linear(list_features)))
            #print('Size of list feature:', list_features.size())
            
            features = torch.cat((user_features,list_features),dim=0)

            features = self.ReLU(self.RGT_layer1(features, edge_index, edge_type))
            features = self.ReLU(self.RGT_layer2(features, edge_index, edge_type))
            
            if self.pretrain == True:
                mask = [1 if i in test_batch.pretrain_test_idx else 0 for i in test_batch.n_id]
                mask = torch.BoolTensor(mask)
                features = self.drop(self.ReLU(self.out(features)))
                mask = mask[:self.args.test_batch_size]
                pred = self.decoder(features)[:self.args.test_batch_size]
                pred = pred[torch.reshape(mask,(mask.size(0),1))]
                label = test_batch.y[:self.args.test_batch_size]
                label = label[mask]
                globals.pre_pred_test.append(pred.squeeze().cpu())
                globals.pre_label_test.append(label.squeeze().cpu())
                
                mse = mean_squared_error(label.cpu(),pred.cpu())

            else:
                mask = [1 if i in test_batch.finetune_test_idx else 0 for i in test_batch.n_id]
                mask = torch.BoolTensor(mask)
                features = self.drop(self.ReLU(self.out(features)))
                mask = mask[:self.args.test_batch_size]
                pred = self.classifier(features)[:self.args.test_batch_size]
                pred = pred[mask.nonzero(as_tuple=True)[0]]
                pred_binary = torch.argmax(pred, dim=1)
                label = test_batch.y[:self.args.test_batch_size]
                label = label[mask]
                
                globals.fine_pred_test += list(pred_binary.squeeze().cpu())
                globals.fine_pred_test_prob += list(pred[:,1].squeeze().cpu())
                globals.fine_label_test += list(label.squeeze().cpu())

                acc = accuracy_score(label.cpu(), pred_binary.cpu())
                f1 = f1_score(label.cpu(), pred_binary.cpu(),average="macro")
                precision =precision_score(label.cpu(), pred_binary.cpu(),average="macro")
                recall = recall_score(label.cpu(), pred_binary.cpu(),average="macro")
                bacc = balanced_accuracy_score(label.cpu(), pred_binary.cpu())
                # auc = roc_auc_score(label.cpu(), pred[:,1].cpu())

                self.log("acc", acc)
                self.log("f1",f1)
                self.log("precision", precision)
                self.log("recall", recall)
                self.log("bacc",bacc)
                #self.log("auc", auc)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2_reg, amsgrad=False)
        scheduler = CosineAnnealingLR(optimizer, T_max=16, eta_min=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }