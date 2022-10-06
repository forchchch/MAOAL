import torch
import torch.nn as nn
import backbone_model

class MTL_model(nn.Module):
    def __init__(self, task_list, feature_dim = 512,backbone='resnet',uncert = False):
        super().__init__()
        if backbone == 'resnet':
            self.base_model = backbone_model.resnet18()
        else:
            self.base_model = backbone_model.Convnet4(feature_dim)
        self.task_num = len(task_list)
        self.heads = nn.ModuleList( [nn.Linear(feature_dim, task_list[i]) for i in range(self.task_num)] )
        if uncert:
            self.logsigma = nn.Parameter(-0.5*torch.ones(1))
            #self.logsigma = nn.Parameter(-0.5*torch.ones(2))
    
    def forward(self, x):
        features = self.base_model(x)
        logits = []
        for i in range(self.task_num):
            logit = self.heads[i](features)
            logits.append(logit)
        return logits