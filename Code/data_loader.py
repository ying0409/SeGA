import torch
from torch_geometric.data import Data

def load_data(args):
    print("loading user features...")
    user_cat_features = torch.load(args.dataset_path + "user_cat_properties_tensor.pt", map_location='cpu')
    user_prop_features = torch.load(args.dataset_path + "user_num_properties_tensor.pt", map_location='cpu')
    user_tweet_features = torch.load(args.dataset_path + "user_tweets_tensor.pt", map_location='cpu')
    user_des_features = torch.load(args.dataset_path + "user_des_tensor.pt", map_location='cpu')
    user_x = torch.cat((user_cat_features, user_prop_features, user_tweet_features, user_des_features), dim=1)
    
    if args.lst:
        print("loading list features...")
        list_cat_features = torch.load(args.dataset_path + "list_cat_properties_tensor.pt", map_location='cpu')
        list_prop_features = torch.load(args.dataset_path + "list_num_properties_tensor.pt", map_location='cpu')
        list_tweet_features = torch.load(args.dataset_path + "list_tweets_tensor.pt", map_location='cpu')
        list_des_features = torch.load(args.dataset_path + "list_des_tensor.pt", map_location='cpu')
        
        # the list embedding is padded to match the dimension of the user embedding
        list_padding = torch.zeros(args.list_num, args.user_numeric_num + args.user_cat_num - args.list_numeric_num - args.list_cat_num)
        list_x = torch.cat((list_cat_features, list_prop_features, list_tweet_features, list_des_features, list_padding), dim=1)
    
    print("loading pretrain label, index...")
    if args.pretext_task == "contrastive":
        if args.template == "l" or args.template == "s":
            pretrain_label = torch.load(args.dataset_path + "pretrain_labels_index.pt", map_location='cpu')
        else:
            pretrain_label = torch.load(args.dataset_path + "pretrain_labels_index_{}.pt".format(args.template), map_location='cpu')
    elif args.pretext_task == "multi":
        pretrain_label = torch.load(args.dataset_path + "pretrain_labels_index_multi.pt", map_location='cpu')
    print("loading finetune label, index...")
    finetune_label = torch.load(args.dataset_path + "finetune_label.pt", map_location='cpu')
    
    if args.lst:
        all_x = torch.cat((user_x,list_x),dim=0)
        print("loading user & list edges...")
        edge_index = torch.load(args.dataset_path + "edge_index(ul_{}).pt".format(args.edge_types), map_location='cpu')
        edge_type = torch.load(args.dataset_path + "edge_type(ul_{}).pt".format(args.edge_types), map_location='cpu').unsqueeze(-1)
        # add label -1 for list node
        lst_label_padding = torch.full((args.list_num,), -1)
        if args.pretext_task == "multi":
            pre_lst_label_padding = torch.full((args.list_num, 153), -1)
            pretrain_label = torch.cat((pretrain_label,pre_lst_label_padding))
        else:
            pretrain_label = torch.cat((pretrain_label,lst_label_padding))
        finetune_label = torch.cat((finetune_label,lst_label_padding))
    else:
        all_x = user_x
        print("loading user edges...")
        edge_index = torch.load(args.dataset_path + "edge_index.pt", map_location='cpu')
        edge_type = torch.load(args.dataset_path + "edge_type.pt", map_location='cpu').unsqueeze(-1)
    
    pretrain_data = Data(x=all_x, edge_index=edge_index, edge_attr=edge_type, y=pretrain_label, finetune_label=finetune_label)
    pretrain_data.pretrain_train_idx = torch.load(args.dataset_path + "finetune_train_idx.pt", map_location='cpu')
    pretrain_data.pretrain_valid_idx = torch.load(args.dataset_path + "finetune_val_idx.pt", map_location='cpu')
    pretrain_data.pretrain_test_idx = torch.load(args.dataset_path + "finetune_test_idx.pt", map_location='cpu')
    print(pretrain_data.pretrain_test_idx)
    # save the original index for all nodes (pick up user nodes during loss calculation)
    pretrain_data.n_id = torch.arange(pretrain_data.num_nodes)

    finetune_data = Data(x=all_x, edge_index=edge_index, edge_attr=edge_type, y=finetune_label, pretrain_label=pretrain_label)
    finetune_data.finetune_train_idx = torch.load(args.dataset_path + "finetune_train_idx.pt", map_location='cpu')
    finetune_data.finetune_valid_idx = torch.load(args.dataset_path + "finetune_val_idx.pt", map_location='cpu')
    finetune_data.finetune_test_idx = torch.load(args.dataset_path + "finetune_test_idx.pt", map_location='cpu')
    print(finetune_data.finetune_test_idx)
    finetune_data.n_id = torch.arange(finetune_data.num_nodes)

    return pretrain_data, finetune_data