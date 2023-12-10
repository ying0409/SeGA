import pandas as pd
from datetime import datetime as dt
import numpy as np
import os
from tqdm import tqdm
from transformers import pipeline
import json
import pytz
import torch
import csv

feature_extract=pipeline('feature-extraction', model='roberta-base', tokenizer='roberta-base', device=0, padding=True, truncation=True, max_length=50, add_special_tokens = True)

def Read_Data():
    print("Read data")
    sample_path = "/home/TwiBot-22/TwiBot-22/sample_100001/"
    path = "/home/TwiBot-22/TwiBot-22/"
    fine_split = pd.read_csv(sample_path+'split.csv')
    fine_label = pd.read_csv(sample_path+'label.csv')
    
    edge = pd.read_csv(path+'edge.csv')
    user = pd.read_json(sample_path+'user.json')
    lst = pd.read_json(path+'list.json')
    hashtag = pd.read_json(path+'hashtag.json')
    
    # set id for user, list, hashtag
    index_uid = user['id']
    uid_index = {uid:index for index,uid in enumerate(index_uid.values)}
    index_lid = lst['id']
    lid_index = {lid:index for index,lid in enumerate(index_lid.values)}
    index_hid = hashtag['id']
    hid_index = {hid:index for index,hid in enumerate(index_hid.values)}
    
    if not os.path.exists('./processed_data/lid_tweet.json') and not os.path.exists('./processed_data/uid_tweet.json'):
        # extract tweets contain in list and tweets post by user
        filter = edge['relation']=='contain'
        sample_edge = edge[filter].reset_index()
        tid_lid = {}
        for index, edge in sample_edge.iterrows():
            tid_lid[edge['target_id']] = edge['source_id']
            lid_tweet = {i:[] for i in range(len(index_lid))}
            uid_tweet = {i:[] for i in range(len(index_uid))}
        for i in range(9):
            name='tweet_'+str(i)+'.json'
            print(name)
            user_tweets=json.load(open(path+name,'r'))
            for each in user_tweets:
                if each['id'] in tid_lid:
                    lid = tid_lid[each['id']]
                    index = lid_index[lid]
                    lid_tweet[index].append(text)
                uid='u'+str(each['author_id'])
                text=each['text']
                try:
                    index=uid_index[uid]
                    uid_tweet[index].append(text)
                except KeyError:
                    continue
        print(lid_tweet)
        json.dump(lid_tweet,open('./processed_data/lid_tweet.json','w'))
        json.dump(uid_tweet,open('./processed_data/uid_tweet.json','w'))

    return fine_label, fine_split, edge, user, lst, hashtag

def Label_Split_Processed(fine_label, pre_split, fine_split, user):
    print('Extracting labels and splits')
    uid_fine_label = {uid:fine_label for uid, fine_label in zip(fine_label['id'].values,fine_label['label'].values)}
    uid_pre_split = {uid:pre_split for uid, pre_split in zip(pre_split['id'].values,pre_split['split'].values)}
    uid_fine_split = {uid:fine_split for uid, fine_split in zip(fine_split['id'].values,fine_split['split'].values)}
    pre_train_idx = []
    pre_test_idx = []
    pre_val_idx = []
    
    fine_train_idx = []
    fine_test_idx = []
    fine_val_idx = []
    fine_label = []
    
    user_idx = user['id']
    print(user_idx)
    for i,uid in enumerate(tqdm(user_idx.values)):
        single_pre_split = uid_pre_split[uid]
        if single_pre_split=='train':
            pre_train_idx.append(i)
        elif single_pre_split=='test':
            pre_test_idx.append(i)
        else:
            pre_val_idx.append(i)

        if uid in uid_fine_split:
            single_fine_split = uid_fine_split[uid]
            if single_fine_split=='train':
                fine_train_idx.append(i)
            elif single_fine_split=='test':
                fine_test_idx.append(i)
            else:
                fine_val_idx.append(i)

        if uid in uid_fine_label:
            single_fine_label = uid_fine_label[uid]
            if single_fine_label =='human':
                fine_label.append(0)
            elif single_fine_label == 'troll':
                fine_label.append(2)
            else:
                fine_label.append(1)
        else:
            fine_label.append(-1)

    pre_train_mask = torch.LongTensor(pre_train_idx)
    pre_valid_mask = torch.LongTensor(pre_val_idx)
    pre_test_mask = torch.LongTensor(pre_test_idx)
    
    fine_train_mask = torch.LongTensor(fine_train_idx)
    fine_valid_mask = torch.LongTensor(fine_val_idx)
    fine_test_mask = torch.LongTensor(fine_test_idx)
    fine_label=torch.LongTensor(fine_label)
    print(pre_train_mask,pre_valid_mask,pre_test_mask)
    print(fine_train_mask,fine_valid_mask,fine_test_mask,fine_label)
    torch.save(pre_train_mask,"./processed_data/pretrain_train_idx.pt")
    torch.save(pre_valid_mask,"./processed_data/pretrain_val_idx.pt")
    torch.save(pre_test_mask,"./processed_data/pretrain_test_idx.pt")
    torch.save(fine_train_mask,"./processed_data/finetune_train_idx.pt")
    torch.save(fine_valid_mask,"./processed_data/finetune_val_idx.pt")
    torch.save(fine_test_mask,"./processed_data/finetune_test_idx.pt")
    torch.save(fine_label,'./processed_data/finetune_label.pt')

def Initial_User_Embedding(user, mask_feature, mask_type):
    print('Initial user embedding')
    print('Extracting num_properties')
    # following count, tweet count, followers count, username length, created at, name length
    following_count=[]
    for i,each in enumerate(user['public_metrics']):
        if i==len(user):
            break
        if each is not None and isinstance(each,dict):
            if each['following_count'] is not None:
                following_count.append(each['following_count'])
            else:
                following_count.append(0)
        else:
            following_count.append(0)
            
    statues=[]
    for i,each in enumerate(user['public_metrics']):
        if i==len(user):
            break
        if each is not None and isinstance(each,dict):
            if each['tweet_count'] is not None:
                statues.append(each['tweet_count'])
            else:
                statues.append(0)
        else:
            statues.append(0)

    followers_count=[]
    for each in user['public_metrics']:
        if each is not None and each['followers_count'] is not None:
            followers_count.append(int(each['followers_count']))
        else:
            followers_count.append(0)
            
    num_username=[]
    for each in user['username']:
        if each is not None:
            num_username.append(len(each))
        else:
            num_username.append(int(0))
            
    created_at=user['created_at']
    created_at=pd.to_datetime(created_at,unit='s')

    date0=dt.strptime('Tue Sep 5 00:00:00 +0000 2020 ','%a %b %d %X %z %Y ')
    active_days=[]
    for each in created_at:
        active_days.append((date0-pytz.UTC.localize(each.replace(tzinfo=None))).days)
        
    active_days=pd.DataFrame(active_days)
    active_days=active_days.fillna(int(1)).astype(np.float32)

    screen_name_length=[]
    for each in user['name']:
        if each is not None:
            screen_name_length.append(len(each))
        else:
            screen_name_length.append(int(0))


    if mask_feature == 'followers_count':
        if mask_type == 'zero':
            followers_count = torch.zeros(len(user),1)
        elif mask_type == 'mean':
            followers_count = pd.DataFrame(followers_count)
            followers_count = torch.full((len(user),1),followers_count.mean().values[0])
        elif mask_type == 'random':
            followers_count = torch.randn(len(user),1)
    else:
        followers_count = pd.DataFrame(followers_count)
        followers_count = (followers_count-followers_count.mean())/followers_count.std()
        followers_count = torch.tensor(np.array(followers_count),dtype=torch.float32)
    
    active_days = pd.DataFrame(active_days)
    active_days.fillna(int(0))
    active_days = active_days.fillna(int(0)).astype(np.float32)

    active_days = (active_days-active_days.mean())/active_days.std()
    active_days = torch.tensor(np.array(active_days),dtype=torch.float32)

    screen_name_length = pd.DataFrame(screen_name_length)
    screen_name_length = (screen_name_length-screen_name_length.mean())/screen_name_length.std()
    screen_name_length = torch.tensor(np.array(screen_name_length),dtype=torch.float32)

    if mask_feature == 'following_count':
        if mask_type == 'zero':
            following_count = torch.zeros(len(user),1)
        elif mask_type == 'mean':
            following_count = pd.DataFrame(following_count)
            following_count = torch.full((len(user),1),following_count.mean().values[0])
        elif mask_type == 'random':
            following_count = torch.randn(len(user),1)
    else:
        following_count = pd.DataFrame(following_count)
        following_count = (following_count-following_count.mean())/following_count.std()
        following_count = torch.tensor(np.array(following_count),dtype=torch.float32)

    statues = pd.DataFrame(statues)
    statues = (statues-statues.mean())/statues.std()
    statues = torch.tensor(np.array(statues),dtype=torch.float32)
    
    num_properties_tensor = torch.cat([followers_count,active_days,screen_name_length,following_count,statues],dim=1)
    print(num_properties_tensor)

    pd.DataFrame(num_properties_tensor.detach().numpy()).isna().value_counts()
    torch.save(num_properties_tensor,'./processed_data/user_num_properties_tensor_{}_{}.pt'.format(mask_feature,mask_type))

    print('Extracting cat_properties')
    # protected, verified, profile image url
    protected=user['protected']
    verified=user['verified']

    protected_list=[]
    for each in protected:
        if each == True:
            protected_list.append(1)
        else:
            protected_list.append(0)
            
    verified_list=[]
    for each in verified:
        if each == True:
            verified_list.append(1)
        else:
            verified_list.append(0)
            
    default_profile_image=[]
    for each in user['profile_image_url']:
        if each is not None:
            if each=='https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png':
                default_profile_image.append(int(1))
            elif each=='':
                default_profile_image.append(int(1))
            else:
                default_profile_image.append(int(0))
        else:
            default_profile_image.append(int(1))

    protected_tensor=torch.tensor(protected_list,dtype=torch.float)
    verified_tensor=torch.tensor(verified_list,dtype=torch.float)
    default_profile_image_tensor=torch.tensor(default_profile_image,dtype=torch.float)

    cat_properties_tensor=torch.cat([protected_tensor.reshape([100001,1]),verified_tensor.reshape([100001,1]),default_profile_image_tensor.reshape([100001,1])],dim=1)

    torch.save(cat_properties_tensor,'./processed_data/user_cat_properties_tensor.pt')

    print('Running user discription embedding')
    path="./processed_data/user_des_tensor.pt"
    user_text=list(user['description'])
    if not os.path.exists(path):
        des_vec=[]
        for k,each in enumerate(tqdm(user_text)):
            if each is None:
                des_vec.append(torch.zeros(768))
            else:
                feature=torch.Tensor(feature_extract(each))
                for (i,tensor) in enumerate(feature[0]):
                    if i==0:
                        feature_tensor=tensor
                    else:
                        feature_tensor+=tensor
                feature_tensor/=feature.shape[1]
                des_vec.append(feature_tensor)
                
        des_tensor=torch.stack(des_vec,0)
        torch.save(des_tensor,path)
    else:
        des_tensor=torch.load(path)
    print('Finished')

def Initial_List_Embedding(lst):
    print('Initial list embedding')
    print('Extracting num_properties')
    # follower count, member count, created at, name length
    follower_count = []
    for each in lst['follower_count']:
        if each is not None:
            follower_count.append(int(each))
        else:
            follower_count.append(0)
    
    member_count = []
    for each in lst['member_count']:
        if each is not None:
            member_count.append(int(each))
        else:
            member_count.append(0)    
    
    num_listname=[]
    for each in lst['name']:
        if each is not None:
            num_listname.append(len(each))
        else:
            num_listname.append(int(0))
    
    created_at=lst['created_at']
    created_at=pd.to_datetime(created_at,unit='s')       
    
    date0=dt.strptime('Tue Sep 5 00:00:00 +0000 2020 ','%a %b %d %X %z %Y ')
    active_days=[]
    for each in created_at:
        active_days.append((date0-pytz.UTC.localize(each.replace(tzinfo=None))).days)
        
    active_days=pd.DataFrame(active_days)
    active_days=active_days.fillna(int(1)).astype(np.float32)
    
    follower_count = pd.DataFrame(follower_count)
    follower_count = (follower_count-follower_count.mean())/follower_count.std()
    follower_count = torch.tensor(np.array(follower_count),dtype=torch.float32)
    
    member_count = pd.DataFrame(member_count)
    member_count = (member_count-member_count.mean())/member_count.std()
    member_count = torch.tensor(np.array(member_count),dtype=torch.float32)
    
    active_days = pd.DataFrame(active_days)
    active_days.fillna(int(0))
    active_days = active_days.fillna(int(0)).astype(np.float32)

    active_days = (active_days-active_days.mean())/active_days.std()
    active_days = torch.tensor(np.array(active_days),dtype=torch.float32)
    
    num_listname=pd.DataFrame(num_listname)
    num_listname=(num_listname-num_listname.mean())/num_listname.std()
    num_listname=torch.tensor(np.array(num_listname),dtype=torch.float32)
    
    num_properties_tensor=torch.cat([follower_count, member_count, active_days, num_listname],dim=1)
    pd.DataFrame(num_properties_tensor.detach().numpy()).isna().value_counts()
    
    print('Extracting cat_properties')
    # private
    private = lst['private']
    private_list = []
    for each in private:
        if each == True:
            private_list.append(1)
        else:
            private_list.append(0)
    private_tensor=torch.tensor(private_list,dtype=torch.float)
    cat_properties_tensor = private_tensor.reshape([21870,1])
    
    torch.save(num_properties_tensor,'./processed_data/list_num_properties_tensor.pt')
    torch.save(cat_properties_tensor,'./processed_data/list_cat_properties_tensor.pt')
    
    print('Running list discription embedding')
    path="./processed_data/list_des_tensor.pt"
    lst_text=list(lst['description'])
    if not os.path.exists(path):
        des_vec=[]
        for k,each in enumerate(tqdm(lst_text)):
            if each is None:
                des_vec.append(torch.zeros(768))
            else:
                feature=torch.Tensor(feature_extract(each))
                for (i,tensor) in enumerate(feature[0]):
                    if i==0:
                        feature_tensor=tensor
                    else:
                        feature_tensor+=tensor
                feature_tensor/=feature.shape[1]
                des_vec.append(feature_tensor)
                
        des_tensor=torch.stack(des_vec,0)
        torch.save(des_tensor,path)
    else:
        des_tensor=torch.load(path)
    print('Running list tweets embedding')
    path="./processed_data/list_tweets_tensor.pt"
    if not os.path.exists(path):
        tweets_list=[]
        each_list_tweets=json.load(open("./processed_data/lid_tweet.json",'r'))
        for i in tqdm(range(len(each_list_tweets))):
            if len(each_list_tweets[str(i)])==0:
                total_each_list_tweets=torch.zeros(768)
            else:
                for j in range(len(each_list_tweets[str(i)])):
                    each_tweet=each_list_tweets[str(i)][j]
                    if each_tweet is None:
                        total_word_tensor=torch.zeros(768)
                    else:
                        each_tweet_tensor=torch.tensor(feature_extract(each_tweet))
                        for k,each_word_tensor in enumerate(each_tweet_tensor[0]):
                            if k==0:
                                total_word_tensor=each_word_tensor
                            else:
                                total_word_tensor+=each_word_tensor
                        total_word_tensor/=each_tweet_tensor.shape[1]
                    if j==0:
                        total_each_list_tweets=total_word_tensor
                    elif j==20:
                        break
                    else:
                        total_each_list_tweets+=total_word_tensor
                if (j==20):
                    total_each_list_tweets/=20
                else:
                    total_each_list_tweets/=len(each_list_tweets[str(i)])
                    
            tweets_list.append(total_each_list_tweets)
                
        tweet_tensor=torch.stack(tweets_list)
        torch.save(tweet_tensor,"./processed_data/list_tweets_tensor.pt")
                    
    else:
        tweets_tensor=torch.load(path)
    print('Finished')
    
def Initial_Hashtag_Embedding(hashtag):
    print("Initial Hashtag Embedding")
    path="./processed_data/hash_name_tensor.pt"
    hashtag = pd.read_json('/home/TwiBot-22/hashtag.json')
    hashtag_name = list(hashtag['tag_name'])
    print(hashtag_name)
    if not os.path.exists(path):
        name_vec = []
        for k,each in enumerate(tqdm(hashtag_name)):
            if each is None:
                name_vec.append(torch.zeros(768))
            else:
                feature = torch.Tensor(feature_extract(each))
                for (i,tensor) in enumerate(feature[0]):
                    if i==0:
                        feature_tensor = tensor
                    else:
                        feature_tensor += tensor
                feature_tensor /= feature.shape[1]
                name_vec.append(feature_tensor)
            
            if k%10000 == 0:
                des_tensor=torch.stack(name_vec,0)
                torch.save(des_tensor,path)
    else:
        print("start from 4190001")
        name_vec = []
        for k,each in enumerate(tqdm(hashtag_name)):
            if k>=4190001:
                if each is None:
                    name_vec.append(torch.zeros(768))
                else:
                    feature = torch.Tensor(feature_extract(each))
                    for (i,tensor) in enumerate(feature[0]):
                        if i==0:
                            feature_tensor = tensor
                        else:
                            feature_tensor += tensor
                    feature_tensor /= feature.shape[1]
                    name_vec.append(feature_tensor)
                if k%10000 == 0:
                    des_tensor=torch.stack(name_vec,0)
                    torch.save(des_tensor,'./processed_data/hash_name_tensor2.pt')
        des_tensor=torch.stack(name_vec,0)
        torch.save(des_tensor,'./processed_data/hash_name_tensor2.pt')
        # des_tensor=torch.load(path)

def Preprocess_Edge(edge,user,lst,hashtag):
    print(edge)
    print('extracting edge_index & edge_type')
    # edge_type = edge['relation'].unique()
    edge_type = ['following','followers','own','membership','followed','discuss']
    rname_index = {rname:index for index,rname in enumerate(edge_type)}
    index_uid = user['id']
    uid_index = {uid:index for index,uid in enumerate(index_uid.values)}
    index_lid = lst['id']
    lid_index = {lid:index for index,lid in enumerate(index_lid.values)}
    index_hid = hashtag['id']
    hid_index = {hid:index for index,hid in enumerate(index_hid.values)}
    edge_index=[]
    edge_type=[]
    
    discuss_edge = pd.read_csv("/home/SEGA-main/Sega/datasets/processed_data/discuss_edge.csv")
    for index,e in tqdm(discuss_edge.iterrows()):
        sid = e['source_id']
        tid = e['target_id']
        edge_index.append([uid_index[sid],len(index_uid)+len(index_lid)+hid_index[tid]])
        edge_type.append(rname_index[e['relation']])
    
    for index,e in tqdm(edge.iterrows()):
        sid = e['source_id']
        tid = e['target_id']
        if e['relation']=='following':
            try:
                edge_index.append([uid_index[sid],uid_index[tid]])
                edge_type.append(rname_index[e['relation']])
            except KeyError:
                continue
        elif e['relation']=='followers':
            try:
                edge_index.append([uid_index[sid],uid_index[tid]])
                edge_type.append(rname_index[e['relation']])
            except KeyError:
                continue
        elif e['relation']=='own':
            try:
                edge_index.append([uid_index[sid],len(index_uid)+lid_index[tid]])
                edge_type.append(rname_index[e['relation']])
            except KeyError:
                continue
        elif e['relation']=='membership':
            try:
                edge_index.append([len(index_uid)+lid_index[sid],uid_index[tid]])
                edge_type.append(rname_index[e['relation']])
            except KeyError:
                continue
        elif e['relation']=='followed':
            try:
                edge_index.append([len(index_uid)+lid_index[sid],uid_index[tid]])
                edge_type.append(rname_index[e['relation']])
            except KeyError:
                continue
    
    torch.save(torch.LongTensor(edge_index).t(),"./processed_data/edge_index(ulh).pt")
    torch.save(torch.LongTensor(edge_type),"./processed_data/edge_type(ulh).pt")

def Pretrain_Label(user, *params):
    print(params)
    public_metrics = ['followers_count','following_count','tweet_count','listed_count']
    pre_label = []
    print(user)
    for i in params:
        print(i)
        print(user['public_metrics'])
        if i in public_metrics:
            for j in user['public_metrics']:
                pre_label.append(j[i])
        else:
            if i in user:
                pre_label.append(user[i])
    pre_label = pd.DataFrame(pre_label)
    print("min, std", pre_label.min().values[0], pre_label.std().values[0])
    pre_label = (pre_label-pre_label.min())/pre_label.std()
    pre_label = torch.tensor(np.array(pre_label),dtype=torch.float32).flatten()
    print(pre_label)
    torch.save(pre_label,'./processed_data/pretrain_label_{}.pt'.format(params[0]))

def ChatGPT_Pretrain_Label(user):
    index_uid = user['id']
    print(len(index_uid))
    uid_index = {uid:index for index,uid in enumerate(index_uid.values)}

    pretrain_labels = [[] for i in range(len(index_uid))]
    multi_label = pd.read_csv('/home/Sega/datasets/processed_data/chatgpt_pretrain_multi_labels.csv',header=None)
    print(multi_label)
    for id,(index,labels) in multi_label.iterrows():
        for label in labels:
            print(label)
            pretrain_labels[uid_index[index]].append(int(label))
    pretrain_labels_index = [torch.tensor(item) for item in pretrain_labels]
    pretrain_labels_index = torch.LongTensor(pretrain_labels_index)
    print(pretrain_labels_index)
    torch.save(pretrain_labels_index,'/home/Sega/datasets/processed_data/pretrain_labels_index_multi.pt')

    # pretrain_labels_index = [-1 for i in range(len(index_uid))]
    # topic_data = pd.read_csv('/home/Sega/datasets/processed_data/chatgpt_pretrain_topic_labels.csv',header=None)
    # print(topic_data)
    # topic_data = topic_data.values.tolist()

    # for (id,t1,t2) in topic_data:
    #     index = t1 * 17 + t2
    #     pretrain_labels_index[uid_index[id]] = index

    # # for (id,t1,e1,t2,e2) in topic_data:
    # #     index = (t1 * 9 + e1) * (9 * 17) + t2 * 9 + e2
    # #     pretrain_labels_index[uid_index[id]] = index
    # #     pretrain_labels[uid_index[id]] = prompt_tensor_dict[index]
    
    # pretrain_labels_index = torch.LongTensor(pretrain_labels_index)
    # print(pretrain_labels_index)
    # torch.save(pretrain_labels_index,'/home/Sega/datasets/processed_data/pretrain_labels_index_topic.pt')

    # pretrain_labels_index = [-1 for i in range(len(index_uid))]
    # emotion_data = pd.read_csv('/home/Sega/datasets/processed_data/chatgpt_pretrain_emotion_labels.csv',header=None)
    # print(emotion_data)
    # emotion_data = emotion_data.values.tolist()

    # for (id,e1,e2) in emotion_data:
    #     index = e1 * 9 + e2
    #     pretrain_labels_index[uid_index[id]] = index
    
    # pretrain_labels_index = torch.LongTensor(pretrain_labels_index)
    # print(pretrain_labels_index)
    # torch.save(pretrain_labels_index,'/home/Sega/datasets/processed_data/pretrain_labels_index_emotion.pt')

    # pretrain_labels_index = [-1 for i in range(len(index_uid))]
    # topic_data = pd.read_csv('/home/Sega/datasets/processed_data/chatgpt_pretrain_topic_labels.csv',header=None)
    # emotion_data = pd.read_csv('/home/Sega/datasets/processed_data/chatgpt_pretrain_emotion_labels.csv',header=None)
    # print(topic_data,emotion_data)
    # topic_data = topic_data.values.tolist()
    # emotion_data = emotion_data.values.tolist()

    # for ((tid,t1,t2),(eid,e1,e2)) in zip(topic_data,emotion_data):
    #     index = (t1 * 17 + t2) * (9 * 9) + e1 * 9 + e2
    #     pretrain_labels_index[uid_index[tid]] = index
    
    # pretrain_labels_index = torch.LongTensor(pretrain_labels_index)
    # print(pretrain_labels_index)
    # torch.save(pretrain_labels_index,'/home/Sega/datasets/processed_data/pretrain_labels_index_topic_emotion.pt')

fine_label, fine_split, edge, user, lst, hashtag = Read_Data()

Initial_User_Embedding(user, 'followers_count', 'random')
Initial_User_Embedding(user, 'following_count', 'random')
Preprocess_Edge(edge, user, lst, hashtag)
Label_Split_Processed(fine_label, pre_split, fine_split, user)
Pretrain_Label(user,'followers_count')
Initial_List_Embedding(lst)
Initial_Hashtag_Embedding(hashtag)
Preprocess_Edge(edge,user,lst,hashtag)
ChatGPT_Pretrain_Label(user)

# pos_prompts = pd.read_csv('/home/SEGA-main/Sega/pos_prompts.csv')
# neg_prompts = pd.read_csv('/home/SEGA-main/Sega/neg_prompts.csv')

# with open('/home/SEGA-main/Sega/pos_prompts.csv') as f:
#     reader = csv.reader(f)
#     pos_prompts = list(reader)
# with open('/home/SEGA-main/Sega/neg_prompts.csv') as f:
#     reader = csv.reader(f)
#     neg_prompts = list(reader)
# pos_prompts = np.array(pos_prompts)
# neg_prompts = np.array(neg_prompts)
# print(pos_prompts.shape)
# print(neg_prompts.shape)
# result = np.concatenate((pos_prompts, neg_prompts), axis=1)
# print(result.shape)
# result = list(result)

