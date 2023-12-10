import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch_geometric
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score, roc_auc_score, mean_squared_error
from os import listdir
import globals
from SeGA import SeGA
from data_loader import load_data
from embedding_visualization import user_embedding_visualization
import numpy as np
import random
import os
import json
import pandas as pd
import torch.nn.functional as F

def build_args():
    parser = argparse.ArgumentParser(description="SeGA")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--dataset_path", type=str, default="/home/yychang/Sega/datasets/processed_data/sample/")
    parser.add_argument("--devices", type=int, default=[0])
    parser.add_argument("--device", type=str, default='cuda:0')

    pretrain = parser.add_mutually_exclusive_group()
    pretrain.add_argument("--pretrain", action="store_true")
    pretrain.add_argument("--no-pretrain", action="store_true")
    lst = parser.add_mutually_exclusive_group()
    lst.add_argument("--lst", action="store_true")
    lst.add_argument("--no-lst", action="store_true")
    
    parser.add_argument("--pretext_task", type=str, default='contrastive') # multi
    parser.add_argument("--template", type=str, default='l') # s, t, e, te, n
    parser.add_argument("--prompt_encoder", type=str, default='SimCSE') # RoBERTa
    parser.add_argument("--neg_num", type=int,default=100)
    parser.add_argument("--node_types", type=str, default='ul') # u
    parser.add_argument("--edge_types", type=str, default='all')
    parser.add_argument("--pretrain_load", action="store_true")
    parser.add_argument("--finetune_load", type=int, default=None)
    
    parser.add_argument("--pretrain_epochs", type=int, default=100)
    parser.add_argument("--finetune_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--test_batch_size", type=int, default=150)
    parser.add_argument("--linear_channels", type=int, default=128)
    parser.add_argument("--out_channel", type=int, default=128)
    parser.add_argument("--user_channel", type=int, default=64)
    parser.add_argument("--trans_head", type=int, default=2)
    parser.add_argument("--semantic_head", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--T", type=int, default=0.1)
    parser.add_argument("--data_loader", type=int, default=4)
    
    parser.add_argument("--user_num", type=int, default=100001, help="number of users")
    parser.add_argument("--user_numeric_num", type=int, default=5, help="number of numerical features of users")
    parser.add_argument("--user_cat_num", type=int, default=3, help="number of catgorical features of users")
    parser.add_argument("--user_des_channel", type=int, default=768, help="number of description channels of users")
    parser.add_argument("--user_tweet_channel", type=int, default=768, help="number of tweet channels of users")
    
    parser.add_argument("--list_num", type=int, default=20788, help="number of lists")
    parser.add_argument("--list_numeric_num", type=int, default=4, help="number of numerical features of lists")
    parser.add_argument("--list_cat_num", type=int, default=1, help="number of catgorical features of lists")
    parser.add_argument("--list_des_channel", type=int, default=768, help="number of description channels of lists")
    parser.add_argument("--list_tweet_channel", type=int, default=768, help="number of tweet channels of lists")
    
    parser.add_argument("--num_edge_type", type=int, default=5, help="number of edge types")
    parser.add_argument("--num_layers", type=int, default=2, help="number of hidden layers")
    parser.add_argument("--num_hidden", type=int, default=256, help="number of hidden units")
    parser.add_argument("--num_out", type=int, default=3, help="number of output dimension")
    parser.add_argument("--pretrain_lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--finetune_lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--l2_reg", type=float, default=1e-4, help="weight decay")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = build_args()

    acc_list = []; precision_list = []; recall_list = []; f1_list = []
    class_acc_list = [[],[],[]]; class_precision_list = [[],[],[]]; class_recall_list = [[],[],[]]; class_f1_list = [[],[],[]]

    for seed in args.seeds:
        globals.pre_pred_test = []
        globals.pre_fine_label_test = []
        globals.pre_pre_label_test = []
        globals.fine_pred_test = []
        globals.fine_pred_test_prob = []
        globals.fine_label_test = []
        globals.all_test_user_embedding = []
        globals.all_fine_user_embedding = []

        # set seeds
        pl.seed_everything(seed, workers=True)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch_geometric.seed_everything(seed)
        random.seed(seed)                                                                                                                                                     
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)

        # set hyper-parameters
        if args.lst:
            args.node_types = "ul"
            args.num_edge_type = 5
            args.semantic_head = 5
        else:
            args.node_types = "u"
            args.num_edge_type = 2
            args.semantic_head = 2

        if args.template == "l" or args.template == "s" or args.template == "te":
            args.neg_num = 100
        elif args.template == "t":
            args.neg_num = 50
        elif args.template == "e":
            args.neg_num = 10

        print("total edge_types:", args.num_edge_type)
        print("total node types:", args.node_types)
        
        # load data
        pretrain_data, finetune_data = load_data(args)
        print(pretrain_data)
        print(finetune_data)

        # set callbacks
        fine_checkpoint_callback = ModelCheckpoint(
            monitor='val_f1',
            mode='max',
            filename='{epoch:02d}-{val_f1:.4f}',
            save_top_k=1,
            verbose=True)
        
        if args.pretrain:
            print("Pretrining...")
            print("loading data...")
            pre_train_loader = NeighborLoader(pretrain_data, num_neighbors=[256]*args.data_loader, input_nodes=pretrain_data.pretrain_train_idx, batch_size=args.batch_size, shuffle=True, num_workers=2)
            pre_valid_loader = NeighborLoader(pretrain_data, num_neighbors=[256]*args.data_loader, input_nodes=pretrain_data.pretrain_valid_idx, batch_size=args.batch_size, num_workers=2)
            pre_test_loader = NeighborLoader(pretrain_data, num_neighbors=[256]*args.data_loader, input_nodes=pretrain_data.pretrain_test_idx, batch_size=args.test_batch_size, num_workers=2)

            # pretrain model
            model = SeGA(args, pretrain=True)
            trainer = pl.Trainer(gpus=args.devices, num_nodes=1, max_epochs=args.pretrain_epochs, precision=16, log_every_n_steps=1)
            trainer.fit(model, pre_train_loader)

            if args.pretext_task == "contrastive":
                torch.save(model.state_dict(), "pretrain_model_{}_{}_{}_{}.pt".format(args.node_types, args.edge_types, args.prompt_encoder, args.template)) # ex: pretrain_model_ul_all_SimCSE.pt
            elif args.pretext_task == "multi":
                torch.save(model.state_dict(), "pretrain_model_{}_{}_{}.pt".format(args.node_types, args.edge_types, args.pretext_task)) # ex: pretrain_model_ul_multi.pt
        
        print("Finetuning...")
        print("loading data...")
        fine_train_loader = NeighborLoader(finetune_data, num_neighbors=[256]*args.data_loader, input_nodes=finetune_data.finetune_train_idx, batch_size=args.batch_size, shuffle=True, num_workers=2)
        fine_valid_loader = NeighborLoader(finetune_data, num_neighbors=[256]*args.data_loader, input_nodes=finetune_data.finetune_valid_idx, batch_size=args.batch_size, num_workers=2)
        fine_test_loader = NeighborLoader(finetune_data, num_neighbors=[256]*args.data_loader, input_nodes=finetune_data.finetune_test_idx, batch_size=args.test_batch_size, num_workers=2)

        model = SeGA(args, pretrain=False)
        
        # load pretrain model
        if args.pretrain or args.pretrain_load:
            if args.pretext_task == "contrastive":
                print("loading pretrain model...","pretrain_model_{}_{}_{}_{}.pt".format(args.node_types, args.edge_types, args.prompt_encoder, args.template))
                model.load_state_dict(torch.load("pretrain_model_{}_{}_{}_{}.pt".format(args.node_types, args.edge_types, args.prompt_encoder, args.template)))
            elif args.pretext_task == "multi":
                print("loading pretrain model...","pretrain_model_{}_{}_{}.pt".format(args.node_types, args.edge_types, args.pretext_task))
                model.load_state_dict(torch.load("pretrain_model_{}_{}_{}.pt".format(args.node_types, args.edge_types, args.pretext_task)))

        # finetune model
        trainer = pl.Trainer(gpus=args.devices, max_epochs=args.finetune_epochs, precision=16, log_every_n_steps=1, num_nodes=1, callbacks=[fine_checkpoint_callback])
        if args.finetune_load == None:
            trainer.fit(model, fine_train_loader, fine_valid_loader)
            dir = './lightning_logs/version_{}/checkpoints/'.format(trainer.logger.version)
            best_path = './lightning_logs/version_{}/checkpoints/{}'.format(trainer.logger.version, listdir(dir)[0])
            json.dump(vars(args),open('lightning_logs/version_{}/arguments.json'.format(trainer.logger.version),'w'))
        else:
            dir = './lightning_logs/version_{}/checkpoints/'.format(str(args.finetune_load))
            print("load: ",listdir(dir)[0])
            best_path = './lightning_logs/version_{}/checkpoints/{}'.format(str(args.finetune_load), listdir(dir)[0])

        # test
        best_model = SeGA.load_from_checkpoint(checkpoint_path=best_path, args=args, pretrain=False)
        trainer.test(best_model, fine_test_loader, verbose=True)
        user_embedding_visualization(np.array(globals.all_fine_user_embedding), np.array(globals.fine_label_test),"SeGA")
        
        # print performance
        acc = accuracy_score(globals.fine_label_test, globals.fine_pred_test)
        f1 = f1_score(globals.fine_label_test, globals.fine_pred_test,average="macro",zero_division=0)
        precision = precision_score(globals.fine_label_test, globals.fine_pred_test,average="macro",zero_division=0)
        recall = recall_score(globals.fine_label_test, globals.fine_pred_test,average="macro",zero_division=0)
        print("overall performance")
        print("acc: {} \t f1: {} \t precision: {} \t recall: {}".format(acc, f1, precision, recall))

        acc_list.append(acc)
        f1_list.append(f1)
        precision_list.append(precision)
        recall_list.append(recall)

        # print performance for each class
        count_class = [[0,0,0] for _ in range(args.num_out)]
        for l_test,p_test in zip(globals.fine_label_test,globals.fine_pred_test):
            count_class[l_test][p_test] += 1
        for c in count_class:
            print(c)

        class_accuracy = [0] * args.num_out
        class_precision = [0] * args.num_out
        class_recall = [0] * args.num_out
        class_f1 = [0] * args.num_out

        class_metrics = []
        for i in range(args.num_out):
            predicted_class = [1 if label == i else 0 for label in globals.fine_pred_test]
            ground_truth_class = [1 if label == i else 0 for label in globals.fine_label_test]

            class_accuracy[i] = accuracy_score(ground_truth_class, predicted_class)
            class_precision[i] = precision_score(ground_truth_class, predicted_class,zero_division=0)
            class_recall[i] = recall_score(ground_truth_class, predicted_class, zero_division=0)
            class_f1[i] = f1_score(ground_truth_class, predicted_class, zero_division=0)
            
            if i==0:
                print("Class: normal user")
            elif i==1:
                print("Class: bot")
            elif i==2:
                print("Class: troll")
            print("acc: {} \t precision:{}\t recall: {}\t f1: {}".format(class_accuracy[i], class_precision[i], class_recall[i], class_f1[i]))
            class_acc_list[i].append(class_accuracy[i])
            class_precision_list[i].append(class_precision[i])
            class_recall_list[i].append(class_recall[i])
            class_f1_list[i].append(class_f1[i])

final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
final_f1, final_f1_std = np.mean(f1_list), np.std(f1_list)
final_precision, final_precision_std = np.mean(precision_list), np.std(precision_list)
final_recall, final_recall_std = np.mean(recall_list), np.std(recall_list)
print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
print(f"# final_f1: {final_f1:.4f}±{final_f1_std:.4f}")
print(f"# final_precision {final_precision:.4f}±{final_precision_std:.4f}")
print(f"# final_recall: {final_recall:.4f}±{final_recall_std:.4f}")

for i in range(args.num_out):
    final_acc, final_acc_std = np.mean(class_acc_list[i]), np.std(class_acc_list[i])
    final_precision, final_precision_std = np.mean(class_precision_list[i]), np.std(class_precision_list[i])
    final_recall, final_recall_std = np.mean(class_recall_list[i]), np.std(class_recall_list[i])
    final_f1, final_f1_std = np.mean(class_f1_list[i]), np.std(class_f1_list[i])
    if i==0:
        print("Class: normal user")
    elif i==1:
        print("Class: bot")
    elif i==2:
        print("Class: troll")
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# final_precision: {final_precision:.4f}±{final_precision_std:.4f}")
    print(f"# final_recall: {final_recall:.4f}±{final_recall_std:.4f}")
    print(f"# final_f1: {final_f1:.4f}±{final_f1_std:.4f}")

