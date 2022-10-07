import torch
import json
from torch.utils.data import Dataset
import os
import numpy as np

def convert(i_feature,u_feature,feature_map):
    all_feature = []
    convert_feature = []
    for i in i_feature:
        all_feature.append(i)
    for u in u_feature:
        all_feature.append(u)
    for m,feature in enumerate(all_feature):
        convert_feature.append(feature_map[m][feature])
    return convert_feature
class movie_dataset(Dataset):
    def __init__(self, set_path, data, fewshot_dict, user_feature, item_feature, feature_map):
        self.set = json.load(open(set_path,'r'))
        self.data = data
        self.fewshot_dict = fewshot_dict
        self.item_feature = item_feature
        self.user_feature = user_feature
        self.feature_map = feature_map
    def __getitem__(self, index):
        pair_id = self.set[index]
        user = self.data[pair_id][0]
        item = self.data[pair_id][1]
        rating = self.data[pair_id][2]
        click = self.data[pair_id][3]
        effect = self.fewshot_dict[str(pair_id)]
        i_feature = self.item_feature[str(item)]
        u_feature = self.user_feature[str(user)]
        all_feature = convert(i_feature,u_feature,self.feature_map)
        
        return np.asarray(all_feature), click, rating, effect
    def __len__(self):
        return len(self.set)

class movie_dataset_naive(Dataset):
    def __init__(self, set_path, data, fewshot_dict, user_feature, item_feature, feature_map):
        self.set = json.load(open(set_path,'r'))
        self.data = data
        self.fewshot_dict = fewshot_dict
        self.item_feature = item_feature
        self.user_feature = user_feature
        self.feature_map = feature_map
        self.pairid_to_id = {}
        n = 0
        for item in self.set:
            self.pairid_to_id[item] = n
            n = n + 1

    def __getitem__(self, index):
        pair_id = self.set[index]
        user = self.data[pair_id][0]
        item = self.data[pair_id][1]
        rating = self.data[pair_id][2]
        click = self.data[pair_id][3]
        effect = self.fewshot_dict[str(pair_id)]
        i_feature = self.item_feature[str(item)]
        u_feature = self.user_feature[str(user)]
        all_feature = convert(i_feature,u_feature,self.feature_map)
        sample_num = self.pairid_to_id[pair_id]
        
        return np.asarray(all_feature), click, rating, effect, sample_num
    def __len__(self):
        return len(self.set)

class amazon_dataset(Dataset):
    def __init__(self, set_path, data, fewshot_dict, item_feature):
        self.set = json.load(open(set_path,'r'))
        self.data = data
        self.fewshot_dict = fewshot_dict
        self.item_feature = item_feature
    def __getitem__(self, index):
        pair_id = self.set[index]
        user = self.data[pair_id][0]
        item = self.data[pair_id][1]
        rating = self.data[pair_id][2]
        click = self.data[pair_id][3]
        effect = self.fewshot_dict[str(pair_id)]
        all_feature = [user, item, self.item_feature[str(item)] ]
        return np.asarray(all_feature), click, rating, effect
    def __len__(self):
        return len(self.set)