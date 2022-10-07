import json
import os

root = "./data/ml-1m/preprocessed_json"
item_path = os.path.join(root, 'item_features_movielen1m.json')
user_path = os.path.join(root, 'user_features_movielen1m.json')
with open(item_path,'r') as f:
    item_features = json.load(f)
with open(user_path, 'r') as f:
    user_features = json.load(f)
print(len(item_features))
print(len(user_features))

item_id_set = []
item_cate_set = []
user_id_set = []
user_gender_set = []
user_job_set = []
user_age_set = []

for item in item_features:
    item_id = item[0]
    item_cate = item[1]
    if item_id not in item_id_set:
        item_id_set.append(item_id)
    if item_cate not in item_cate_set:
        item_cate_set.append(item_cate)

for user in user_features:
    user_id = user[0]
    user_gender = user[1]
    user_job = user[2]
    user_age = user[3]
    if user_id not in user_id_set:
        user_id_set.append(user_id)
    if user_gender not in user_gender_set:
        user_gender_set.append(user_gender)
    if user_job not in user_job_set:
        user_job_set.append(user_job)
    if user_age not in user_age_set:
        user_age_set.append(user_age)

all_features = [item_id_set,item_cate_set,user_id_set, user_gender_set,user_job_set, user_age_set]
all_fields =  [len(m) for m in all_features]
all_dict = []
for set_item in all_features:
    tmp_dict = {}
    for i, item in enumerate(set_item):
        tmp_dict[item] = i
    print(len(tmp_dict))
    all_dict.append(tmp_dict)

print(all_fields)
with open( os.path.join(root,'feature_mapping.json'),'w' ) as f:
    json.dump(all_dict,f)
with open( os.path.join(root,'all_fields.json'),'w' ) as f:
    json.dump(all_fields,f)

user_feature_dict = {}
item_feature_dict = {}
for feature in user_features:
    user_id = feature[0]
    user_feature_dict[user_id] = feature
for feature in item_features:
    item_id = feature[0]    
    item_feature_dict[item_id] = feature

with open( os.path.join(root,'item_feature.json'),'w' ) as f:
    json.dump(item_feature_dict,f)
with open( os.path.join(root,'user_feature.json'),'w' ) as f:
    json.dump(user_feature_dict,f)
print(len(user_feature_dict))
print(len(item_feature_dict))