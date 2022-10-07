# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 17:56:01 2021

@author: luoh1
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionInteract(nn.Module):
    """
        多头注意力的交互层
    """
    
    def __init__(self, embed_size, head_num, dropout, residual = True):
        """
        """
        super(MultiHeadAttentionInteract,self).__init__()
        self.embed_size = embed_size
        self.head_num = head_num
        self.dropout = dropout
        self.use_residual = residual
        self.attention_head_size = embed_size // head_num
            
        # 直接定义参数, 更加直观
        
        self.W_Q = nn.Parameter(torch.Tensor(embed_size, embed_size))
        self.W_K = nn.Parameter(torch.Tensor(embed_size, embed_size))
        self.W_V = nn.Parameter(torch.Tensor(embed_size, embed_size))
        
        if self.use_residual:
            self.W_R = nn.Parameter(torch.Tensor(embed_size, embed_size))
        
        # 初始化, 避免计算得到nan
        for weight in self.parameters():
            nn.init.xavier_uniform_(weight)
        
    
    def forward(self, x):
        """
            x : (batch_size, feature_fields, embed_dim)
        """
        
        # 线性变换到注意力空间中
        Query = torch.tensordot(x, self.W_Q, dims=([-1], [0]))
        Key = torch.tensordot(x, self.W_K, dims=([-1], [0]))
        Value = torch.tensordot(x, self.W_V, dims=([-1], [0]))
        
        # Head (head_num, bs, fields, D / head_num)
        Query = torch.stack(torch.split(Query, self.attention_head_size, dim = 2))
        Key = torch.stack(torch.split(Key, self.attention_head_size, dim = 2))
        Value = torch.stack(torch.split(Value, self.attention_head_size, dim = 2))
        
        # 计算内积
        inner = torch.matmul(Query, Key.transpose(-2,-1))
        inner = inner / self.attention_head_size ** 0.5
        
        # Softmax归一化权重
        attn_w = F.softmax(inner, dim=-1)
        attn_w = F.dropout(attn_w, p = self.dropout)
        
        # 加权求和
        results = torch.matmul(attn_w, Value)
        
        # 拼接多头空间
        results = torch.cat(torch.split(results, 1, ), dim = -1)
        results = torch.squeeze(results, dim = 0) # (bs, fields, D)
        
        # 残差学习(resnet YYDS?)
        if self.use_residual:
            results = results + torch.tensordot(x, self.W_R, dims=([-1], [0]))
        
        results = F.relu(results)
        
        return results


class AutoIntbackbone(nn.Module):
    """
            Automatic Feature Interaction Net
    """        
    def __init__(self, feature_fields, embed_dim, head_num, 
                 attn_layers, mlp_dims, dropout):
    
        super(AutoIntbackbone, self).__init__()
        self.feature_fields = feature_fields
        self.offsets = np.array((0, *np.cumsum(feature_fields)[:-1]), dtype = np.long)
        
        # Embedding layer
        self.embedding = nn.Embedding(sum(feature_fields)+1, embed_dim)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

        # MultiHeadAttetion layer
        self.atten_output_dim = len(feature_fields) * embed_dim
        attns = []
        for i in range(attn_layers):
            attns.append(MultiHeadAttentionInteract(embed_size=embed_dim, 
                                                    head_num = head_num, 
                                                    dropout = dropout))
        self.attns = nn.Sequential(*attns)
    
    def forward(self, x):
        """
            x : (batch_size, num_fileds)
        """
        tmp = x + x.new_tensor(self.offsets).unsqueeze(0)
        # embeded dense vector
        embeded_x = self.embedding(tmp)
        att_feature = self.attns(embeded_x).view(-1, self.atten_output_dim)

        return att_feature

class AutoIntNet(nn.Module):
    """
            Automatic Feature Interaction Net
    """        
    def __init__(self, feature_fields, embed_dim, head_num, 
                 attn_layers, mlp_dims, dropout,uncert=False, task_num=1):
    
        super(AutoIntNet, self).__init__()

        feature_dim = len(feature_fields) * embed_dim
        self.backbone = AutoIntbackbone(feature_fields, embed_dim, head_num, attn_layers, mlp_dims, dropout)
        self.head = nn.Linear(feature_dim, 1) 
    
    def forward(self, x):
        features = self.backbone(x)
        logit = self.head(features)
        return logit.view(-1), features
    
        
        
        
        
        