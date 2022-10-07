import codecs
import json
item_feature_path = './data/ml-1m/movies.dat'
user_feature_path = './data/ml-1m/users.dat'

item_features = []
f1=codecs.open(item_feature_path,'r',encoding = "ISO-8859-1")

a = 0
for line in f1.readlines():
    m = line.strip().split("::")
    item_features.append([m[0],m[2].strip().split("|")[0]])
    a = a + 1
with open('./data/ml-1m/preprocessed_json/item_features_movielen1m.json','w') as f_obj1:
    json.dump(item_features,f_obj1)  

user_features = []
f2=open(user_feature_path,'r')
a = 0
for line in f2.readlines():
    m = line.strip().split("::")
    print(m)
    user_features.append([m[0],m[1],m[2],m[3]])
    a = a + 1
with open('./data/ml-1m/preprocessed_json/user_features_movielen1m.json','w') as f_obj2:
    json.dump(user_features,f_obj2)  