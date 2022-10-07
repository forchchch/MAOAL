import json
rating_path = './data/ml-1m/ratings.dat'
needed_rate = []
all_users = []
all_items = []
m = 0
with open(rating_path, 'r') as f:
    for line in f.readlines():
        rate_info = line.split("::")
        user = int(rate_info[0])
        item = int(rate_info[1])
        if user not in all_users:
            all_users.append(user)
        if item not in all_items:
            all_items.append(item)
        rating = int(rate_info[2])
        time = int(rate_info[3][:-1])
        if rating !=3:
            if rating > 3:
                click = 1
                needed_rate.append([user,item,rating,click,time])
            if rating < 3:
                click = 0 
                needed_rate.append([user,item,rating,click,time])
        m = m + 1
        if m%5000==0:
            print(m)
print("rating num:",len(needed_rate))
print("user num:", len(all_users))
print("item num:", len(all_items))

all_set_id = [m for m in range(len(needed_rate))]
import random
random.shuffle(all_set_id)
all_len = len(needed_rate)
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1
all_train_set_id = all_set_id[0:int(train_ratio*all_len)]
valid_set_id = all_set_id[int(train_ratio*all_len):int(train_ratio*all_len)+int(valid_ratio*all_len)]
test_set_id = all_set_id[int(train_ratio*all_len)+int(valid_ratio*all_len):]
print(len(all_train_set_id))
print(len(valid_set_id))
print(len(test_set_id))

import numpy as np
aux_shot = 1
aux_set_id = []
rest_train_set = []
aux_num = 512
shot_recorder = np.zeros(6040)
for order in all_train_set_id:
    user = needed_rate[order][0]
    total_num = shot_recorder.sum()
    if shot_recorder[user-1]<aux_shot and total_num<aux_num:
        aux_set_id.append(order)
        shot_recorder[user-1] += 1
    else:
        rest_train_set.append(order)
print(len(aux_set_id))
print(len(rest_train_set))

fewshot_dict = {}
for num in all_set_id:
    fewshot_dict[num] = 0.0
fewshot = 18
fewshotset = []
shot_recorder = np.zeros(6040)
for order in rest_train_set:
    user = needed_rate[order][0]
    if shot_recorder[user-1]<fewshot:
        fewshotset.append(order)
        shot_recorder[user-1] += 1
        fewshot_dict[order] = 1.0
        
for order in aux_set_id:
    fewshotset.append(order)
    fewshot_dict[order]=1.0
print(len(fewshotset))

import os
out_root = './data/ml-1m/preprocessed_json'
if not os.path.exists(out_root):
    os.makedirs(out_root)
with open(os.path.join(out_root,'all_info.json'),'w') as f:
    json.dump(needed_rate,f)
    
with open(os.path.join(out_root,'all_train_set.json'), 'w') as f:
    json.dump(all_train_set_id,f)
with open(os.path.join(out_root,'valid_set.json'), 'w') as f:
    json.dump(valid_set_id,f)
with open(os.path.join(out_root,'test_set.json'), 'w') as f:
    json.dump(test_set_id,f)
with open(os.path.join(out_root,'aux_set.json'), 'w') as f:
    json.dump(aux_set_id,f)
with open(os.path.join(out_root,'rest_train_set.json'), 'w') as f:
    json.dump(rest_train_set,f)
with open(os.path.join(out_root,'fewshot_set.json'), 'w') as f:
    json.dump(fewshotset,f)
with open(os.path.join(out_root,'fewshot_dict.json'), 'w') as f:
    json.dump(fewshot_dict,f)

print(len(all_train_set_id))
print(len(valid_set_id))
print(len(test_set_id))
print(len(aux_set_id))
print(len(rest_train_set))
print(len(fewshot_dict))