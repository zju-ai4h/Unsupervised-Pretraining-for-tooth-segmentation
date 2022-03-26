#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: util
@Time: 4/5/19 3:47 PM
"""


import numpy as np
import torch
import torch.nn.functional as F
import logging

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)
    
    if smoothing:
        eps = 0.2
        n_class = pred.size(1)
        
      
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1).long(), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

        
def load_state_with_same_shape(model, weights):
    model_state = model.state_dict()
    if list(weights.keys())[0].startswith('module.'):
        logging.info("Loading multigpu weights with module. prefix...")
        weights = {k.partition('module.')[2]:weights[k] for k in weights.keys()}

    if list(weights.keys())[0].startswith('encoder.'):
        logging.info("Loading multigpu weights with encoder. prefix...")
        weights = {k.partition('encoder.')[2]:weights[k] for k in weights.keys()}

    filtered_weights = {
        k: v for k, v in weights.items() if k in model_state  and v.size() == model_state[k].size() 
    }
    logging.info("Loading weights:" + ', '.join(filtered_weights.keys()))
    return filtered_weights
