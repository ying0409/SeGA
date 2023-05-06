import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from RGT import RGTPretrain
import RGT
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score, roc_auc_score, mean_squared_error
from os import listdir
import globals

def load_data(args):
    print("loading user features...")
    user_cat_features = torch.load(args.dataset_path + "user_cat_properties_tensor.pt", map_location='cpu')
    user_prop_features = torch.load(args.dataset_path + "user_num_properties_tensor.pt", map_location='cpu')
    user_tweet_features = torch.load(args.dataset_path + "user_tweets_tensor.pt", map_location='cpu')
    user_des_features = torch.load(args.dataset_path + "user_des_tensor.pt", map_location='cpu')
    user_x = torch.cat((user_cat_features, user_prop_features, user_tweet_features, user_des_features), dim=1)
    
    print("loading list features...")
    list_cat_features = torch.load(args.dataset_path + "list_cat_properties_tensor.pt", map_location='cpu')
    list_prop_features = torch.load(args.dataset_path + "list_num_properties_tensor.pt", map_location='cpu')
    list_tweet_features = torch.load(args.dataset_path + "list_tweets_tensor.pt", map_location='cpu')
    list_des_features = torch.load(args.dataset_path + "list_des_tensor.pt", map_location='cpu')
    list_padding = torch.zeros(21870, args.user_numeric_num + args.user_cat_num - args.list_numeric_num - args.list_cat_num)
    list_x = torch.cat((list_cat_features, list_prop_features, list_tweet_features, list_des_features, list_padding), dim=1)
    
    x = torch.cat((user_x,list_x),dim=0)
    
    print("loading edges...")
    edge_index = torch.load(args.dataset_path + "edge_index.pt", map_location='cpu')
    edge_type = torch.load(args.dataset_path + "edge_type.pt", map_location='cpu').unsqueeze(-1)
    
    print("loading pretrain label, index...")
    pretrain_label = torch.load(args.dataset_path + "pretrain_label.pt", map_location='cpu')
    padding = torch.full((args.list_num,), -1)
    pretrain_label = torch.cat((pretrain_label,padding))
    pretrain_data = Data(x=x, edge_index = edge_index, edge_attr=edge_type, y=pretrain_label)
    pretrain_data.pretrain_train_idx = torch.load(args.dataset_path + "pretrain_train_idx.pt", map_location='cpu')
    pretrain_data.pretrain_valid_idx = torch.load(args.dataset_path + "pretrain_val_idx.pt", map_location='cpu')
    pretrain_data.pretrain_test_idx = torch.load(args.dataset_path + "pretrain_test_idx.pt", map_location='cpu')
    pretrain_data.n_id = torch.arange(pretrain_data.num_nodes)

    print("loading finetune label, index...")
    finetune_label = torch.load(args.dataset_path + "finetune_label.pt", map_location='cpu')
    finetune_label = torch.cat((finetune_label,padding))
    finetune_data = Data(x=x, edge_index = edge_index, edge_attr=edge_type, y=finetune_label)
    finetune_data.finetune_train_idx = torch.load(args.dataset_path + "finetune_train_idx.pt", map_location='cpu')
    finetune_data.finetune_valid_idx = torch.load(args.dataset_path + "finetune_val_idx.pt", map_location='cpu')
    finetune_data.finetune_test_idx = torch.load(args.dataset_path + "finetune_test_idx.pt", map_location='cpu')
    finetune_data.n_id = torch.arange(finetune_data.num_nodes)

    return pretrain_data, finetune_data

def build_args():
    parser = argparse.ArgumentParser(description="SeGA")
    parser.add_argument("--seeds", type=int, nargs="+", default=14)
    parser.add_argument("--dataset_path", type=str, default="/home/yychang/SEGA-main/Sega/datasets/processed_data/")
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--pretrain_epochs", type=int, default=1, help="pretrin epochs")
    parser.add_argument("--finetune_epochs", type=int, default=5, help="finetune epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("--test_batch_size", type=int, default=200, help="test batch size")
    parser.add_argument("--linear_channels", type=int, default=128, help="linear channels")
    parser.add_argument("--out_channel", type=int, default=128, help="description channel")
    parser.add_argument("--trans_head", type=int, default=2, help="description channel")
    parser.add_argument("--semantic_head", type=int, default=2, help="description channel")
    parser.add_argument("--dropout", type=float, default=0.3, help="description channel")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pretrain", action="store_true")
    group.add_argument("--no-pretrain", action="store_true")
    
    parser.add_argument("--user_num", type=int, default=1000000, help="number of users")
    parser.add_argument("--user_numeric_num", type=int, default=4, help="user numerical features")
    parser.add_argument("--user_cat_num", type=int, default=3, help="user catgorical features")
    parser.add_argument("--user_des_channel", type=int, default=768, help="user description channel")
    parser.add_argument("--user_tweet_channel", type=int, default=768, help="user tweet channel")
    
    parser.add_argument("--list_num", type=int, default=21870, help="number of lists")
    parser.add_argument("--list_numeric_num", type=int, default=4, help="list numerical features")
    parser.add_argument("--list_cat_num", type=int, default=1, help="list catgorical features")
    parser.add_argument("--list_des_channel", type=int, default=768, help="list description channel")
    parser.add_argument("--list_tweet_channel", type=int, default=768, help="list tweet channel")
    
    parser.add_argument("--num_edge_type", type=int, default=5, help="total edge type")
    
    parser.add_argument("--num_heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num_hidden", type=int, default=256,
                        help="number of hidden units")
    parser.add_argument("--num_out", type=int, default=3,
                        help="number of output dimension")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--l2_reg", type=float, default=1e-5, help="description channel")

    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = build_args()
    if args.seeds != None:
        pl.seed_everything(args.seeds)
        
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        filename='{val_acc:.4f}',
        save_top_k=1,
        verbose=True)

    pretrain_data, finetune_data = load_data(args)
    
    if args.pretrain:
        print("Pretrining...")
        print("loading data...")
        pre_train_loader = NeighborLoader(pretrain_data, num_neighbors=[4]*4, input_nodes=pretrain_data.pretrain_train_idx, batch_size=args.batch_size, shuffle=True)
        pre_valid_loader = NeighborLoader(pretrain_data, num_neighbors=[4]*4, input_nodes=pretrain_data.pretrain_valid_idx, batch_size=args.batch_size)# , num_workers=1
        pre_test_loader = NeighborLoader(pretrain_data, num_neighbors=[4]*4, input_nodes=pretrain_data.pretrain_test_idx, batch_size=args.test_batch_size)# , num_workers=1
    
        model = RGTPretrain(args,pretrain = True)
    
        trainer = pl.Trainer(gpus=1, num_nodes=1, max_epochs=args.pretrain_epochs, precision=16, log_every_n_steps=1)#, callbacks=[checkpoint_callback])
    
        trainer.fit(model, pre_train_loader)#, valid_loader)
        torch.save(model.state_dict(), "model.pt")

        dir = './lightning_logs/version_{}/checkpoints/'.format(trainer.logger.version)
        best_path = 'lightning_logs/version_{}/checkpoints/{}'.format(trainer.logger.version, listdir(dir)[0])

        best_model = RGTPretrain.load_from_checkpoint(checkpoint_path=best_path, args=args, pretrain=True)
        trainer.test(best_model, pre_test_loader, verbose=True)
        
        pred_test = torch.cat(globals.pre_pred_test).cpu()
        label_test = torch.cat(globals.pre_label_test).cpu()
        
        mse = mean_squared_error(label_test.cpu(),pred_test.cpu())
        print("acc: {}".format(mse))
    
    print("Finetuning...")
    print("loading data...")
    fine_train_loader = NeighborLoader(finetune_data, num_neighbors=[256]*4, input_nodes=finetune_data.finetune_train_idx, batch_size=args.batch_size, shuffle=True)
    fine_valid_loader = NeighborLoader(finetune_data, num_neighbors=[256]*4, input_nodes=finetune_data.finetune_valid_idx, batch_size=args.batch_size)# , num_workers=1
    fine_test_loader = NeighborLoader(finetune_data, num_neighbors=[256]*4, input_nodes=finetune_data.finetune_test_idx, batch_size=args.test_batch_size)# , num_workers=1

    model = RGTPretrain(args,pretrain = False)
    if args.pretrain:
        model.load_state_dict(torch.load("model.pt"))

    trainer = pl.Trainer(gpus=1, num_nodes=1, max_epochs=args.finetune_epochs, precision=16, log_every_n_steps=1)#, callbacks=[checkpoint_callback])

    trainer.fit(model, fine_train_loader)#, valid_loader)

    dir = './lightning_logs/version_{}/checkpoints/'.format(trainer.logger.version)
    best_path = './lightning_logs/version_{}/checkpoints/{}'.format(trainer.logger.version, listdir(dir)[0])

    best_model = RGTPretrain.load_from_checkpoint(checkpoint_path=best_path, args=args, pretrain=False)
    trainer.test(best_model, fine_test_loader, verbose=True)
    
    acc = accuracy_score(globals.fine_label_test, globals.fine_pred_test)
    f1 = f1_score(globals.fine_label_test, globals.fine_pred_test,average="macro")
    precision =precision_score(globals.fine_label_test, globals.fine_pred_test,average="macro")
    recall = recall_score(globals.fine_label_test, globals.fine_pred_test,average="macro")
    bacc = balanced_accuracy_score(globals.fine_label_test, globals.fine_pred_test)
    
    print("acc: {} \t f1: {} \t precision: {} \t recall: {} \t balance accuracy: {}".format(acc, f1, precision, recall, bacc))