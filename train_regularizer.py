import torch
import json
from dataset import movie_dataset
import os
from torch.utils.data import DataLoader
from models import Autoint_reg
from sklearn.metrics import roc_auc_score,mean_squared_error
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np
import argparse
import random
from gauxlearn.optim import MetaOptimizer
from weighting_way import GradCosine

def obtain_regularizer(model, add_reg):
    L1_loss = 0.0
    L2_loss = 0.0
    if add_reg:
        for param in model.parameters():
            #L1_loss += torch.sum( torch.abs(param) )
            L2_loss += torch.sum( param*param )
    return  [L2_loss]

def map_param_to_block(shared_params, level):
    param_to_block = {}
    if level == 'param_wise':
        for i, (name, param) in enumerate(shared_params):
            param_to_block[i] = i
        module_num = len(param_to_block)
        return param_to_block, module_num
    
    elif level == 'module_wise':
        for i, (name, param) in enumerate(shared_params):
            if i==0:
                param_to_block[i] = 0
            elif i<=4:
                param_to_block[i] = 1
            elif i<=8:
                param_to_block[i] = 2
            elif i<=12:
                param_to_block[i] = 3
            elif i<=16:
                param_to_block[i] = 4
            else:
                param_to_block[i] = 5
        module_num = 6
        return param_to_block, module_num

class hypermodel(nn.Module):
    def __init__(self, task_num, module_num, param_to_block):
        super(hypermodel, self).__init__()
        self.task_num = task_num
        self.module_num = module_num
        self.param_to_block = param_to_block
        self.modularized_lr = nn.Parameter(torch.ones(task_num, module_num))
        self.nonlinear = nn.ReLU()
        self.scale_factor = 1.0
    def forward(self,loss_vector,shared_params, whether_single=1, train_lr=1.0):
        if whether_single == 1:
            grads = torch.autograd.grad(loss_vector[0], shared_params, create_graph= True)
            if self.nonlinear is not None:
                grads = tuple( self.nonlinear( self.modularized_lr[0][self.param_to_block[m]])*g*train_lr for m,g in enumerate(grads) )
            else:
                grads = tuple( self.modularized_lr[0][ self.param_to_block[m]]*g*train_lr for m,g in enumerate(grads) )
            return grads
        else:
            grads = torch.autograd.grad(loss_vector[0], shared_params, create_graph= True)
            loss_num = len(loss_vector)
            for task_id in range(1, loss_num):
                aux_grads = torch.autograd.grad(loss_vector[task_id], shared_params, create_graph= True)
                if self.nonlinear is not None:
                    grads = tuple( ( g + self.scale_factor*self.nonlinear(self.modularized_lr[task_id-1][self.param_to_block[m]])*g_aux )*train_lr for m,(g,g_aux) in enumerate( zip(grads, aux_grads) ) )  
                else:
                    grads = tuple( ( g + self.scale_factor*self.modularized_lr[task_id-1][self.param_to_block[m]]*g_aux )*train_lr for m,(g,g_aux) in enumerate( zip(grads, aux_grads) ) )  
            return grads

def load_model_test(model,opt,load_dir,test_loader,device):
    info = torch.load(os.path.join(load_dir,'best_predictor.pth'))
    model.load_state_dict(info['model'])
    opt.load_state_dict(info['opt'])
    test_rmse= evaluate_model(model, test_loader, device)
    return test_rmse

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def load_config(init_config):
    with open('./config/params_reg.json', 'r') as f:
        config = json.load(f)
    for key,value in init_config.items():
        config[key] = value
    return config

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def evaluate_model(model, val_loader, device):
    score_true = []
    score_predict = []
    model.eval()
    for data in val_loader:
        feature, click, rate, effect = data
        feature = feature.to(device)
        click = click.to(device).float()
        rate = rate.to(device).float()
        logits,_ = model(feature)
        score_true.extend(rate.tolist())
        score_predict.extend(logits.tolist())
    rmse = np.sqrt(mean_squared_error(score_true,score_predict))
    model.train()
    return rmse

def save_model_and_hyperparameters(model,opt,modular_lr,meta_opt):
    state = { 'model':model.state_dict(), 'opt':opt.state_dict(),'modular_lr':modular_lr.state_dict(),"meta_opt":meta_opt.meta_optimizer.state_dict()}
    torch.save(state,os.path.join(path_check,'best_predictor.pth'))
    return

def train_model(model,epochs,train_loader,val_loader,opt,criterion,criterion2,device,task_num):
    logger.info(f"start standard training")
    model = model.to(device)
    init_rmse = evaluate_model(model,val_loader,device)
    logger.info(f"init rmse:{init_rmse:.6f} ")
    best_rmse = 100.0
    best_epoch = 0
    for epoch in range(epochs):
        ite = 0
        for k,data in enumerate(train_loader):
            model.train()
            feature, click, rate, effect = data
            feature = feature.to(device)
            click = click.to(device).float()
            rate = rate.to(device).float()
            effect = effect.to(device).float()
            logits,_ = model(feature)
            rate_loss = criterion2(logits, rate)
            if config["fewshot"] == 1:
                rate_loss_mean = (rate_loss*effect).sum()/(effect.sum()+1e-12)
            else:
                rate_loss_mean = rate_loss.mean()
            aux_loss = obtain_regularizer(model,True)
            loss_list = [rate_loss_mean, config["main"]["aux_weight"]*aux_loss[0]]
            loss_vec = torch.stack(loss_list).reshape(1,-1)
            total_loss = torch.sum(loss_vec)
            #print(total_loss)
            opt.zero_grad()
            total_loss.backward()
            opt.step()
            if k%100 == 0:
                logger.info(f"epoch:{epoch},iteration: {k}, training loss: {total_loss.item()}, loss vector: {loss_vec}" )
        rmse = evaluate_model(model,val_loader,device)
        if rmse < best_rmse:
            best_rmse = rmse
            save_model(model,opt)
            best_epoch = epoch
        print("current rmse:",rmse, "best rmse:", best_rmse)
        logger.info(f"epoch:{epoch}, current rmse: {rmse:.6f}, best rmse: {best_rmse:.6f}, best epoch:{best_epoch}" )
    return

def train_GCS(model, epochs, train_loader, val_loader, opt, criterion,cirterion2,device, task_num):
    logger.info(f"start GCS training")
    model = model.to(device)
    shared_parameter = [param for name, param in model.named_parameters()]
    gcs_model = GradCosine(0)
    init_rmse = evaluate_model(model,val_loader,device)
    logger.info(f"init rmse:{init_rmse:.6f} ")
    best_rmse = 100.0
    best_epoch = 0
    model.train()
    for epoch in range(epochs):
        for k,data in enumerate(train_loader):
            model.train()
            feature, click, rate, effect = data
            feature = feature.to(device)
            click = click.to(device).float()
            rate = rate.to(device).float()
            effect = effect.to(device).float()
            logits,_ = model(feature)
            rate_loss = criterion2(logits, rate)
            if config["fewshot"] == 1:
                rate_loss_mean = (rate_loss*effect).sum()/(effect.sum()+1e-12)
            else:
                rate_loss_mean = rate_loss.mean()
            aux_loss = obtain_regularizer(model,True)
            loss_list = [rate_loss_mean, config["main"]["aux_weight"]*aux_loss[0]]
            opt.zero_grad()
            gcs_model.backward(loss_list,shared_parameter)
            opt.step()
            if k%100 == 0:
                logger.info(f"epoch:{epoch},iteration: {k}, loss vector: {loss_list}" )
        rmse = evaluate_model(model,val_loader,device)
        if rmse < best_rmse:
            best_rmse = rmse
            save_model(model,opt)
            best_epoch = epoch
        print("current rmse:",rmse, "best rmse:", best_rmse)
        logger.info(f"epoch:{epoch}, current rmse: {rmse:.6f}, best rmse: {best_rmse:.6f}, best epoch:{best_epoch}" )
    return

def modularized_lr_MTL_implicit(model,epochs,train_loader,train_loader2, val_loader,aux_loader,opt,criterion,criterion2,device,task_num):
    logger.info(f"start implicit modularized lr training")
    model = model.to(device)
    shared_parameter = [param for name, param in model.named_parameters()]
    shared_parameter1 = [(name,param) for name, param in model.named_parameters()]
    param_to_block, module_num = map_param_to_block(shared_parameter1, config['level'])
    print("the initial number of modules:",module_num)
    modular = hypermodel(1,module_num,param_to_block)
    #m_optimizer = optim.Adam( modular.parameters(), lr = config['hyper']['lr'],  weight_decay = config['hyper']['decay'] )
    m_optimizer = optim.SGD( modular.parameters(), lr = config['hyper']['lr'], momentum = 0.0, weight_decay = config['hyper']['decay'] )
    meta_optimizer = MetaOptimizer(meta_optimizer= m_optimizer, hpo_lr = 1.0, truncate_iter = 3, max_grad_norm = 10)
    modular = modular.to(device)
    aux_loader_iter = iter(aux_loader)
    train_loader2_iter = iter(train_loader2)
    init_rmse = evaluate_model(model,val_loader,device)
    logger.info(f"init rmse:{init_rmse:.6f} ")
    best_rmse = 100.0
    best_epoch = 0
    model.train()
    counter = 0
    for epoch in range(epochs):
        for k,data in enumerate(train_loader):
            model.train()
            feature, click, rate, effect = data
            feature = feature.to(device)
            click = click.to(device).float()
            rate = rate.to(device).float()
            effect = effect.to(device).float()
            logits,_ = model(feature)
            rate_loss = criterion2(logits, rate)
            if config["fewshot"] == 1:
                rate_loss_mean = (rate_loss*effect).sum()/(effect.sum()+1e-12)
            else:
                rate_loss_mean = rate_loss.mean()
            auxloss = obtain_regularizer(model, True)
            loss_list = [rate_loss_mean, config['main']["aux_weight"]*auxloss[0]]
            common_grads = modular(loss_list, shared_parameter, whether_single =0) 
            loss_vec = torch.stack(loss_list)
            total_loss = torch.sum(loss_vec)
            opt.zero_grad()
            total_loss.backward()
            for p, g in zip(shared_parameter, common_grads):
                p.grad = g
            opt.step()
            del common_grads

            counter += 1
            #print(modular.modularized_lr)
            if k%100 == 0:
                logger.info(f"epoch:{epoch},iteration: {k}, training loss: {total_loss.item()}, loss vector: {loss_vec}" )
            
            if counter % config['interval'] == 0 and epoch > config['pre']:
                try: 
                    meta_feature,meta_click,meta_rate,meta_effect = next(aux_loader_iter)
                except StopIteration:
                    aux_loader_iter = iter(aux_loader)
                    meta_feature,meta_click,meta_rate,meta_effect = next(aux_loader_iter)
                try: 
                    train_feature,train_click,train_rate,train_effect = next(train_loader2_iter)
                except StopIteration:
                    train_loader2_iter = iter(train_loader2)
                    train_feature,train_click,train_rate,train_effect = next(train_loader2_iter)
                
                meta_feature = meta_feature.to(device)
                meta_effect = meta_effect.to(device).float()
                meta_click = meta_click.to(device).float()
                meta_rate = meta_rate.to(device).float()
                meta_prediction,_ = model(meta_feature)
                meta_main_loss = criterion2(meta_prediction, meta_rate)
                meta_total_loss = meta_main_loss.mean()
                
                #meta_total_loss += 0.0*criterion(meta_prediction[1], meta_click).mean()
                # print("before:",modular.modularized_lr)
                
                train_feature = train_feature.to(device)
                train_effect = train_effect.to(device).float()
                train_click = train_click.to(device).float()
                train_rate = train_rate.to(device).float()
                train_logits,_ = model(train_feature)
                train_main_loss = criterion2(train_logits, train_rate)
                if config['fewshot']:
                    train_main_loss = train_main_loss*train_effect
                    train_main_loss_mean = (train_main_loss).sum()/(train_effect.sum() + 1e-12)
                else:
                    train_main_loss_mean = train_main_loss.mean()

                train_loss_vector = []
                train_loss_vector.append(train_main_loss_mean)
                train_auxloss = obtain_regularizer(model, True)
                train_loss_vector.append(config["main"]["aux_weight"]*train_auxloss[0])         
                #train_loss_vector.append(config["main"]["aux_weight"]*train_auxloss[1])                 
                train_common_grads = modular(train_loss_vector, shared_parameter, whether_single =0, train_lr = 1.0) 
                meta_optimizer.step(val_loss=meta_total_loss,train_grads=train_common_grads,aux_params = list(modular.parameters()),shared_parameters = shared_parameter)
                logger.info(f"epoch:{epoch} ,iteration:{k}, main loss:{train_main_loss_mean.item():.6f},meta loss:{meta_total_loss.item():.6f}")
        logger.info(f"modular lr:{modular.nonlinear(modular.modularized_lr)}")
        rmse = evaluate_model(model,val_loader,device)
        if rmse < best_rmse:
            best_rmse = rmse
            save_model_and_hyperparameters(model,opt,modular,meta_optimizer)
            best_epoch = epoch
        print("current rmse:",rmse, "best rmse:", best_rmse)
        logger.info(f"epoch:{epoch}, current rmse: {rmse:.6f}, best rmse: {best_rmse:.6f}, best epoch:{best_epoch}" )
    return

def save_model(model,opt):
    state = { 'model':model.state_dict(), 'opt':opt.state_dict() }
    torch.save(state,os.path.join(path_check,'best_predictor.pth'))
    return
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default="debug", help='experiment name')
args = parser.parse_args()
init_config = vars(args)
config = load_config(init_config)
set_seed(config['seed'])

setting = 'full'
if config['fewshot'] == 1:
    setting = 'fewshot'

path_log = os.path.join('./rec_reg_records','log',setting, args.exp_name)
path_check = os.path.join('./rec_reg_records','checkpoint',setting, args.exp_name)

if not os.path.exists(path_log):
    os.makedirs(path_log)
if not os.path.exists(path_check):
    os.makedirs(path_check)

logger = get_logger(os.path.join(path_log, 'logging.txt'))
logger.info(config)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


root = "./data/ml-1m/preprocessed_json"
if config["use_aux"] == 0:
    train_path = os.path.join(root, 'all_train_set.json')
else:
    train_path = os.path.join(root, 'rest_train_set.json')
    aux_path = os.path.join(root, 'aux_set.json')

data_path = os.path.join(root, 'all_info.json')
fewshot_path = os.path.join(root, 'fewshot_dict.json')
user_feature_path = os.path.join(root, 'user_feature.json')
item_feature_path = os.path.join(root, 'item_feature.json')
feature_map_path = os.path.join(root, 'feature_mapping.json')
all_fileds_path = os.path.join(root, 'all_fields.json')
val_path = os.path.join(root, 'valid_set.json')
test_path = os.path.join(root, 'test_set.json')

with open(data_path, 'r') as f:
    data = json.load(f)
with open(fewshot_path, 'r') as f:
    fewshot_dict = json.load(f)
with open(user_feature_path, 'r') as f:
    user_feature = json.load(f)
with open(item_feature_path,'r') as f:
    item_feature = json.load(f)
with open(feature_map_path, 'r') as f:
    feature_map = json.load(f)
with open(all_fileds_path, 'r') as f:
    all_fields = json.load(f)

train_set = movie_dataset(train_path, data, fewshot_dict, user_feature, item_feature, feature_map)
train_loader = DataLoader(train_set, shuffle=True,num_workers=1, batch_size = config["train_batchsize"])

val_set = movie_dataset(val_path, data, fewshot_dict, user_feature, item_feature, feature_map)
val_loader = DataLoader(val_set, shuffle=False,num_workers=1, batch_size = config["test_batchsize"])

test_set = movie_dataset(test_path, data, fewshot_dict, user_feature, item_feature, feature_map)
test_loader = DataLoader(test_set, shuffle=False,num_workers=1, batch_size = config["test_batchsize"])

if config["use_aux"] == 1:
    train_set2 = movie_dataset(train_path, data, fewshot_dict, user_feature, item_feature, feature_map)
    train_loader2 = DataLoader(train_set2, shuffle=True,num_workers=1, batch_size = config["train_batchsize"])
    aux_set = movie_dataset(aux_path, data, fewshot_dict, user_feature, item_feature, feature_map)
    aux_loader = DataLoader(aux_set, shuffle=True,num_workers=1, batch_size = config["train_batchsize"])    

criterion = nn.BCELoss(reduction = 'none')
criterion2 = nn.MSELoss(reduction = 'none')


if config["mode"] == "common": 
    model = Autoint_reg.AutoIntNet(all_fields, embed_dim = 16, head_num = 4, attn_layers=4, mlp_dims=[16,16], dropout=0.0)
    opt = optim.Adam(model.parameters(), config['main']['lr'])
    train_model(model, config['epochs'], train_loader, val_loader,opt,criterion, criterion2,device, config['task_num'])
if config["mode"] == "GCS":
    model = Autoint_reg.AutoIntNet(all_fields, embed_dim = 16, head_num = 4, attn_layers=4, mlp_dims=[16,16], dropout=0.0)
    opt = optim.Adam(model.parameters(), config['main']['lr'])
    train_GCS(model, config['epochs'], train_loader, val_loader,opt,criterion, criterion2,device, config['task_num'])
if config["mode"] == "modular":
    model = Autoint_reg.AutoIntNet(all_fields, embed_dim = 16, head_num = 4, attn_layers=4, mlp_dims=[16,16], dropout=0.0)
    opt = optim.Adam(model.parameters(), config['main']['lr'])
    modularized_lr_MTL_implicit(model, config['epochs'], train_loader, train_loader2, val_loader, aux_loader, opt, criterion, criterion2,device, config['task_num'])

rmse = load_model_test(model,opt,path_check,test_loader,device)
logger.info( f"test rmse: {rmse:.6f}" )








