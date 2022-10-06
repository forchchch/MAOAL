import torch
import json
import numpy as np
import argparse
import os
from dataset import CIFAR100MTL
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random
import logging
from multitask_model import MTL_model
from AuxLearn.hypernet import MonoHyperNet, MonoLinearHyperNet, MonoNonlinearHyperNet
from AuxLearn.optim import MetaOptimizer_old
from gauxlearn.optim import MetaOptimizer
import torch.optim as optim
import torch.nn as nn
from weighting_way import GradCosine

def map_param_to_block(shared_params, level):
    param_to_block = {}
    if level == 'param_wise':
        for i, (name, param) in enumerate(shared_params):
            param_to_block[i] = i
        module_num = len(param_to_block)
        return param_to_block, module_num
    
    elif level == 'module_wise':
        for i, (name, param) in enumerate(shared_params):
            if name.startswith('base_model.conv1'):
                param_to_block[i] = 0
            elif name.startswith('base_model.conv2'):
                param_to_block[i] = 1
            elif name.startswith('base_model.conv3'):
                param_to_block[i] = 2
            elif name.startswith('base_model.conv4'):
                param_to_block[i] = 3
            else:
                param_to_block[i] = 4
        module_num = 5
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
        #self.scale_factor = 1.0/task_num
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
def load_config(init_config):
    with open('./config/params.json', 'r') as f:
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

def evaluate_model(model,test_loader,device):
    model.eval()
    t_correct = 0.0
    t_total = 0.0
    for data in test_loader:
        picture, label = data
        model = model.to(device)
        picture = picture.to(device)
        label = label.to(device)
        logits = model(picture)
        main_task_logits = logits[0]
        predictions = torch.argmax(main_task_logits,1)
        correct_num = torch.sum(predictions == label) + 0.0
        total_num = len(label)
        t_correct += correct_num
        t_total += total_num + 0.0
    acc = t_correct/t_total
    model.train()
    return acc

def save_model(model,opt):
    state = { 'model':model.state_dict(), 'opt':opt.state_dict() }
    #torch.save(state,'./checkponit/best_predictor.pth')
    torch.save(state,os.path.join(path_check,'best_predictor.pth'))
    return

def save_model_and_hyperparameters(model,opt,modular_lr,meta_opt):
    state = { 'model':model.state_dict(), 'opt':opt.state_dict(),'modular_lr':modular_lr.state_dict(),"meta_opt":meta_opt.meta_optimizer.state_dict()}
    #torch.save(state,'./checkponit/best_predictor.pth')
    torch.save(state,os.path.join(path_check,'best_predictor.pth'))
    return

def load_model_test(model,opt,load_dir,test_loader,device):
    info = torch.load(os.path.join(load_dir,'best_predictor.pth'))
    model.load_state_dict(info['model'])
    opt.load_state_dict(info['opt'])
    test_acc = evaluate_model(model, test_loader, device)
    return test_acc

def train_model(model,epochs,train_loader, auxloaders,opt,criterion,device,test_loader):
    logger.info(f"start standard training")
    train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, config['epochs'], eta_min=0, last_epoch=- 1)
    auxlen = len(auxloaders)
    model = model.to(device)
    best_acc = 0.0
    for epoch in range(epochs):
        ite = 0
        aux_datasets = [iter(auxloader) for auxloader in auxloaders]
        for k, data in enumerate(train_loader):
            model.train()
            loss_list = []
            picture,label = data
            picture = picture.to(device)
            label = label.to(device)
            logits = model(picture)
            tar_logit = logits[0]
            tar_loss = criterion(tar_logit,label)
            loss_list.append(tar_loss)
            for m in range(auxlen):
                a_pic,a_label = aux_datasets[m].next()
                a_pic = a_pic.to(device)
                a_label = a_label.to(device)
                a_logits = model(a_pic)
                aux_loss = criterion(a_logits[m], a_label)
                loss_list.append(config['main']['aux_weight']*aux_loss)
            loss_vec = torch.stack(loss_list)
            all_loss = torch.sum(loss_vec)
            opt.zero_grad()
            all_loss.backward()
            opt.step()
            if (k+1)%50 == 0:
                running_loss = all_loss.item()                         
                predictions = torch.argmax(tar_logit,1)
                correct_num = torch.sum(predictions == label) + 0.0
                total_num = len(label)
                logger.info(f"epoch:{epoch} ,iteration:{k+1}, training loss:{running_loss:.6f}, training acc:{correct_num/total_num:.6f}"  )
        acc = evaluate_model(model,test_loader,device)
        if acc>best_acc:
            best_acc = acc
            save_model(model,opt)
        print("current accuacy:",acc,"best accuracy:",best_acc)
        logger.info(f"epoch:{epoch},current accuacy: {acc:.6f}, best accuracy: {best_acc:.6f}" )
        train_scheduler.step()
    return

def modularized_lr_MTL_implicit(model,epochs,train_loader, auxloaders,opt,criterion,device,test_loader,meta_loader):
    logger.info(f"start implicit modularized lr training")
    model = model.to(device)
    train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, config['epochs'], eta_min=0, last_epoch=- 1)
    shared_parameter = [param for name, param in model.named_parameters() if 'heads' not in name]
    shared_parameter1 = [(name,param) for name, param in model.named_parameters() if 'heads' not in name]
    param_to_block, module_num = map_param_to_block(shared_parameter1, config['level'])
    print("the initial number of modules:",module_num)
    auxlen = len(auxloaders)
    modular = hypermodel(auxlen,module_num,param_to_block)
    #modular = hypermodel_all_task(1,module_num,param_to_block)
    #m_optimizer = optim.Adam( modular.parameters(), lr = 1e-2, weight_decay = config['hyper']['decay'] )
    m_optimizer = optim.SGD( modular.parameters(), lr = config['hyper']['lr'], momentum = 0.9, weight_decay = config['hyper']['decay'] )
    meta_optimizer = MetaOptimizer(meta_optimizer= m_optimizer, hpo_lr = config["hpolr"], truncate_iter = 3, max_grad_norm = 10)
    modular = modular.to(device)
    meta_loader_iter = iter(meta_loader)
    best_acc = 0.0
    counter = 0
    for epoch in range(epochs):
        ite = 0
        aux_datasets = [iter(auxloader) for auxloader in auxloaders]
        for data in train_loader:
            model.train()
            loss_list = []
            picture,label = data
            picture = picture.to(device)
            label = label.to(device)
            logits = model(picture)
            tar_logit = logits[0]
            tar_loss = criterion(tar_logit,label)
            loss_list.append(tar_loss)
            try: 
                for m in range(auxlen):
                    a_pic,a_label = aux_datasets[m].next()
                    a_pic = a_pic.to(device)
                    a_label = a_label.to(device)
                    a_logits = model(a_pic)
                    aux_loss = criterion(a_logits[m], a_label)
                    loss_list.append(config['main']['aux_weight']*aux_loss)
            except StopIteration:
                aux_datasets = [iter(auxloader) for auxloader in auxloaders]
                for m in range(auxlen):
                    a_pic,a_label = aux_datasets[m].next()
                    a_pic = a_pic.to(device)
                    a_label = a_label.to(device)
                    a_logits = model(a_pic)
                    aux_loss = criterion(a_logits[m], a_label)
                    loss_list.append(config['main']['aux_weight']*aux_loss)
            common_grads = modular(loss_list, shared_parameter, whether_single =0) 
            loss_vec = torch.stack(loss_list)
            #total_loss = loss_vec[0] + torch.mean(loss_vec[1:])
            total_loss = torch.sum(loss_vec)
            opt.zero_grad()
            total_loss.backward()
            for p, g in zip(shared_parameter, common_grads):
                p.grad = g
            opt.step()
            del common_grads
            _,predictions = torch.max(logits[0],1)
            correct_num = torch.sum(predictions == label) + 0.0
            total_num = len(label)
            ite += 1
            counter += 1
            #print(modular.modularized_lr)
            if ite % 20 == 0:
                logger.info(f"epoch:{epoch} ,iteration:{ite}, task loss sum:{total_loss.item():.6f}, single loss:{tar_loss.item():.6f},training acc:{correct_num/total_num:.6f}"  )
            
            if counter % config['interval'] == 0 and epoch > config['pre']:
                try: 
                    meta_picture,meta_label = next(meta_loader_iter)
                except StopIteration:
                    meta_loader_iter = iter(meta_loader)
                    meta_picture,meta_label = next(meta_loader_iter)

                
                meta_picture = meta_picture.to(device)
                meta_label = meta_label.to(device)
                meta_prediction = model(meta_picture)
                meta_main_loss = criterion(meta_prediction[0], meta_label)
                # print("before:",modular.modularized_lr)
                t_loss_list = []
                for t_data in train_loader:
                    t_picture, t_label = t_data
                    t_picture = t_picture.to(device)
                    t_label = t_label.to(device)
                    t_logits = model(t_picture)
                    t_tar_logit = t_logits[0]
                    t_tar_loss = criterion(t_tar_logit,t_label)
                    t_loss_list.append(t_tar_loss)
                    for tm in range(auxlen):
                        t_a_pic,t_a_label = aux_datasets[tm].next()
                        t_a_pic = t_a_pic.to(device)
                        t_a_label = t_a_label.to(device)
                        t_a_logits = model(t_a_pic)
                        t_aux_loss = criterion(t_a_logits[tm], t_a_label)
                        t_loss_list.append(config['main']['aux_weight']*t_aux_loss)
                    break
                train_common_grads = modular(t_loss_list, shared_parameter, whether_single =0, train_lr = 1.0) 
                meta_optimizer.step(val_loss=meta_main_loss,train_grads=train_common_grads,aux_params = list(modular.parameters()),shared_parameters = shared_parameter)
                logger.info(f"epoch:{epoch} ,iteration:{ite}, main loss:{t_tar_loss.item():.6f},meta loss:{meta_main_loss.item():.6f}")
        logger.info(f"modular lr:{modular.nonlinear(modular.modularized_lr)}")
        acc = evaluate_model(model,test_loader,device)
        if acc>best_acc:
            best_acc = acc
            save_model_and_hyperparameters(model,opt,modular,meta_optimizer)
        print("current accuacy:",acc,"best accuracy:",best_acc)
        logger.info(f"epoch:{epoch},current accuacy: {acc:.6f}, best accuracy: {best_acc:.6f}" )     
        train_scheduler.step()
    return

def train_aux_model(model,epochs,train_loader, auxloaders,opt,criterion,device,test_loader,meta_loader):
    logger.info(f"start implicit auxLearn")
    model = model.to(device)
    train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, config['epochs'], eta_min=0, last_epoch=- 1)
    auxlen = len(auxloaders)
    loss_combiner = MonoNonlinearHyperNet(0,auxlen+1,[5,5])
    loss_combiner.clamp()
    m_optimizer = optim.SGD( loss_combiner.parameters(), lr = 1e-3, momentum = 0.9, weight_decay = config['hyper']['decay'] )
    meta_optimizer = MetaOptimizer_old(meta_optimizer= m_optimizer, hpo_lr = 1e-4, truncate_iter = 3, max_grad_norm = 10)
    shared_parameter = [param for name, param in model.named_parameters() if 'heads' not in name]
    loss_combiner = loss_combiner.to(device)
    meta_loader_iter = iter(meta_loader)
    best_acc = 0.0
    counter = 0
    for epoch in range(epochs):
        ite = 0
        aux_datasets = [iter(auxloader) for auxloader in auxloaders]
        for data in train_loader:
            model.train()
            loss_list = []
            picture,label = data
            picture = picture.to(device)
            label = label.to(device)
            logits = model(picture)
            tar_logit = logits[0]
            tar_loss = criterion(tar_logit,label)
            loss_list.append(tar_loss)
            for m in range(auxlen):
                a_pic,a_label = aux_datasets[m].next()
                a_pic = a_pic.to(device)
                a_label = a_label.to(device)
                a_logits = model(a_pic)
                aux_loss = criterion(a_logits[m], a_label)
                loss_list.append(config['main']['aux_weight']*aux_loss)
            loss_vec = torch.stack(loss_list).reshape(1,-1)
            total_loss = loss_combiner(loss_vec, tar_loss)
            opt.zero_grad()
            total_loss.backward()
            opt.step()
            _,predictions = torch.max(logits[0],1)
            correct_num = torch.sum(predictions == label) + 0.0
            total_num = len(label)
            ite += 1
            counter += 1
            if ite % 20 == 0:
                logger.info(f"epoch:{epoch} ,iteration:{ite}, task loss sum:{total_loss.item():.6f}, single loss:{tar_loss.item():.6f},training acc:{correct_num/total_num:.6f}"  )
            
            if counter % config['interval'] == 0:
                try: 
                    meta_picture,meta_label = next(meta_loader_iter)
                except StopIteration:
                    meta_loader_iter = iter(meta_loader)
                    meta_picture,meta_label = next(meta_loader_iter)

                
                meta_picture = meta_picture.to(device)
                meta_label = meta_label.to(device)
                meta_prediction = model(meta_picture)
                meta_main_loss = criterion(meta_prediction[0], meta_label)
                for i in range(auxlen):
                    meta_main_loss += 0.0*criterion(meta_prediction[i], meta_label).mean()
                # print("before:",modular.modularized_lr)
                t_train_loss = 0.0
                for t_data in train_loader:
                    t_loss_list = []
                    t_picture, t_label = t_data
                    t_picture = t_picture.to(device)
                    t_label = t_label.to(device)
                    t_logits = model(t_picture)
                    t_tar_logit = t_logits[0]
                    t_tar_loss = criterion(t_tar_logit,t_label)
                    t_loss_list.append(t_tar_loss)
                    for tm in range(auxlen):
                        t_a_pic,t_a_label = aux_datasets[tm].next()
                        t_a_pic = t_a_pic.to(device)
                        t_a_label = t_a_label.to(device)
                        t_a_logits = model(t_a_pic)
                        t_aux_loss = criterion(t_a_logits[tm], t_a_label)
                        t_loss_list.append(config['main']['aux_weight']*t_aux_loss)
                    t_loss_vec = torch.stack(t_loss_list).reshape(1,-1)
                    t_train_loss += loss_combiner(t_loss_vec, t_tar_loss)
                    break
                meta_optimizer.step(
                    val_loss=meta_main_loss,
                    train_loss=t_train_loss,
                    aux_params = list(loss_combiner.parameters()),
                    parameters = shared_parameter
                )
                loss_combiner.clamp()
                logger.info(f"epoch:{epoch} ,iteration:{ite}, main loss:{t_train_loss.item():.6f},meta loss:{meta_main_loss.item():.6f}")
        #logger.info(f"modular lr:{F.relu(modular.modularized_lr)}")
        acc = evaluate_model(model,test_loader,device)
        if acc>best_acc:
            best_acc = acc
            save_model_and_hyperparameters(model,opt,loss_combiner,meta_optimizer)
        print("current accuacy:",acc,"best accuracy:",best_acc)
        logger.info(f"epoch:{epoch},current accuacy: {acc:.6f}, best accuracy: {best_acc:.6f}" )     
        train_scheduler.step()
    return

def train_GCS(model,epochs,train_loader, auxloaders,opt,criterion,device,test_loader):
    logger.info(f"start GCS training")
    model = model.to(device)
    train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, config['epochs'], eta_min=0, last_epoch=- 1)
    shared_parameter = [param for name, param in model.named_parameters() if 'heads' not in name]
    gcs_model = GradCosine(0)
    auxlen = len(auxloaders)
    model = model.to(device)
    best_acc = 0.0
    for epoch in range(epochs):
        ite = 0
        aux_datasets = [iter(auxloader) for auxloader in auxloaders]
        for k, data in enumerate(train_loader):
            model.train()
            loss_list = []
            picture,label = data
            picture = picture.to(device)
            label = label.to(device)
            logits = model(picture)
            tar_logit = logits[0]
            tar_loss = criterion(tar_logit,label)
            loss_list.append(tar_loss)
            for m in range(auxlen):
                a_pic,a_label = aux_datasets[m].next()
                a_pic = a_pic.to(device)
                a_label = a_label.to(device)
                a_logits = model(a_pic)
                aux_loss = criterion(a_logits[m], a_label)
                loss_list.append(config['main']['aux_weight']*aux_loss)
            opt.zero_grad()
            gcs_model.backward(loss_list,shared_parameter)
            opt.step()
            _,predictions = torch.max(logits[0],1)
            correct_num = torch.sum(predictions == label) + 0.0
            total_num = len(label)
            ite += 1
            if ite % 20 == 0:
                logger.info(f"epoch:{epoch} ,iteration:{ite}, single loss:{tar_loss.item():.6f},training acc:{correct_num/total_num:.6f}"  )
        acc = evaluate_model(model,test_loader,device)
        if acc>best_acc:
            best_acc = acc
            save_model(model,opt)
        print("current accuacy:",acc,"best accuracy:",best_acc)
        logger.info(f"epoch:{epoch},current accuacy: {acc:.6f}, best accuracy: {best_acc:.6f}" )     
        train_scheduler.step()
    return
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default="cifar100_5class", help='experiment name')
args = parser.parse_args()
init_config = vars(args)
config = load_config(init_config)
set_seed(config['seed'])


path_log = os.path.join('./records','log', args.exp_name)
path_check = os.path.join('./records','checkpoint', args.exp_name)

if not os.path.exists(path_log):
    os.makedirs(path_log)
if not os.path.exists(path_check):
    os.makedirs(path_check)
logger = get_logger(os.path.join(path_log, 'logging.txt'))
logger.info(config)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

trans_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
])

trans_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
])
aux_subtask = [i for i in range(10)]
target_subtask = 14
task_list = [5]*11
tar_trainset = CIFAR100MTL(root='./data', train=True, transform=trans_train, subset_id=target_subtask)
train_len = len(tar_trainset)
meta_len = int(config['m_ratio']*train_len)
if config['use_aux']:
    tar_trainset, tar_metaset = torch.utils.data.random_split(tar_trainset, (train_len - meta_len, meta_len))
    meta_loader = torch.utils.data.DataLoader(dataset=tar_metaset, batch_size=config['aux_batchsize'], shuffle=False, num_workers=2)

aux_sets = [CIFAR100MTL(root='./data', train=True, transform=trans_train, subset_id=i) for i in aux_subtask]
tar_testset = CIFAR100MTL(root='./data', train=False, transform=trans_test, subset_id=target_subtask)
tar_trainloader = torch.utils.data.DataLoader(dataset=tar_trainset, batch_size=config['train_batchsize'], shuffle=True, num_workers=2)
tar_testloader = torch.utils.data.DataLoader(dataset=tar_testset, batch_size=config['test_batchsize'], shuffle=False, num_workers=2)
auxloaders = [torch.utils.data.DataLoader(dataset=aux_set, batch_size=config['train_batchsize'], shuffle=True, num_workers=2) for aux_set in aux_sets]

if config['mode'] == 'common':
    print("here we enter the normal learning")
    mymodel = MTL_model(task_list, config['feature_dim'])
    crit = nn.CrossEntropyLoss()
    main_optimizer = optim.SGD(mymodel.parameters(), lr=config['main']['lr'], momentum=0.9,weight_decay=config['main']['decay'])
    train_model(mymodel, config['epochs'], tar_trainloader, auxloaders,  main_optimizer, crit, device, tar_testloader)
    test_acc = load_model_test(mymodel, main_optimizer,path_check, tar_testloader, device)
if config['mode'] == "modular":
    print("here we enter the modularized lr learning")
    mymodel = MTL_model(task_list, config['feature_dim'])
    crit = nn.CrossEntropyLoss()
    main_optimizer = optim.SGD(mymodel.parameters(), lr=config['main']['lr'],momentum=0.9, weight_decay=config['main']['decay'])
    modularized_lr_MTL_implicit(mymodel, config['epochs'], tar_trainloader, auxloaders,  main_optimizer, crit, device, tar_testloader,meta_loader)
    test_acc = load_model_test(mymodel, main_optimizer,path_check,tar_testloader, device)
if config['mode'] == "aux":
    mymodel = MTL_model(task_list, config['feature_dim'])
    crit = nn.CrossEntropyLoss()
    main_optimizer = optim.SGD(mymodel.parameters(), lr=config['main']['lr'], momentum=0.9, weight_decay=config['main']['decay'])
    print("here we enter the aux learning")
    train_aux_model(mymodel, config['epochs'], tar_trainloader, auxloaders,  main_optimizer, crit, device, tar_testloader,meta_loader)
    test_acc = load_model_test(mymodel, main_optimizer,path_check,tar_testloader, device)
if config['mode'] == 'GCS':
    print("here we enter the GCS learning")
    mymodel = MTL_model(task_list, config['feature_dim'])
    crit = nn.CrossEntropyLoss()
    main_optimizer = optim.SGD(mymodel.parameters(), lr=config['main']['lr'],momentum=0.9, weight_decay=config['main']['decay'])
    train_GCS(mymodel, config['epochs'], tar_trainloader, auxloaders,  main_optimizer, crit, device, tar_testloader)
    test_acc = load_model_test(mymodel, main_optimizer,path_check,tar_testloader, device)
logger.info( f"best test accuacy: {test_acc:.6f}" )




