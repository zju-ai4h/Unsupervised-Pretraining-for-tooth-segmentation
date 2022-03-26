#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Code based on https://github.com/AnTao97/dgcnn.pytorch/blob/master/model.py
"""

import os
from re import X
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=25, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature
    
class Transform_Net(nn.Module):
    def __init__(self, num_features):
        super(Transform_Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(num_features * 2, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.Mish())
        self.conv2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.Mish())
        self.conv3 = nn.Sequential(nn.Conv1d(256, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.Mish())

        self.linear1 = nn.Linear(1024, 512, bias=True)
        self.linear2 = nn.Linear(512, 256, bias=True)

        self.transform = nn.Linear(256, num_features * num_features, bias=True)
        nn.init.constant_(self.transform.weight, 0)
        nn.init.eye_(self.transform.bias.view(num_features, num_features))
        self.num_features = num_features

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       
        x = self.conv2(x)                       
        x = x.max(dim=-1, keepdim=False)[0]     

        x = self.conv3(x)                       
        x = x.max(dim=-1, keepdim=False)[0]     

        x = F.mish(self.linear1(x))     
        x = F.mish(self.linear2(x))     

        x = self.transform(x)                   
        x = x.view(batch_size, self.num_features, self.num_features)            

        return x

class DGCNN(nn.Module):
    def __init__(self, args):
        super(DGCNN, self).__init__()
        self.k = args.k
        
        self.transform_net = Transform_Net(15)
        # Block 1
        self.conv1 = nn.Sequential(nn.Conv2d(15*2, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.Mish())
        self.conv2_1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.Mish())
        self.conv2_2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.Mish())

        # Block 2
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.Mish())
        self.conv4_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.Mish())
        self.conv4_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.Mish())

        # Block 3
        self.conv5_1 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.Mish())

        self.conv5_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.Mish())


        self.conv6 = nn.Sequential(nn.Conv1d(192, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.Mish())

        self.projector = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(512),
                                    nn.Mish(),
                                    nn.Conv1d(512, 32, kernel_size=1, bias=False))


    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        idx_xyz = knn(x[:, :3, :], self.k)
        x0 = get_graph_feature(x, k=self.k, idx=idx_xyz) 
        t = self.transform_net(x0)              
        x = x.transpose(2, 1)                   
        x = torch.bmm(x, t)                     
        x = x.transpose(2, 1)                   
        
        x = get_graph_feature(x, k=self.k)      
        x = self.conv1(x)                      
        x = self.conv2_1(x)                       
        x = self.conv2_2(x)
        x1 = x.max(dim=-1, keepdim=False)[0]    

        x = get_graph_feature(x1, k=self.k)     
        x = self.conv3(x)                       
        x = self.conv4_1(x)                       
        x = self.conv4_2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]    

        x = get_graph_feature(x2, k=self.k)     
        x = self.conv5_1(x)                       
        x = self.conv5_2(x)
        x3 = x.max(dim=-1, keepdim=False)[0]    

        x = torch.cat((x1, x2, x3), dim=1)      

        x = self.conv6(x)                       
        x = x.max(dim=-1, keepdim=True)[0]      

        x = x.repeat(1, 1, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)

        x = self.projector(x)

        return x