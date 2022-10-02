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

def get_iso_label(res):
    def convert(x):
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33][x]
    return np.array(list(map(convert, res)))
def calculate_shape_IoU(pred_np, seg_np, cat):
    # pred_np [N,]          --smoothed segmentation label for each point in a point cloud
    # seg_np  [N,]          --true segmentation label for each point in a point cloud
    # cat   str             --category( 'u' or 'l')

    if cat == 'u':
        parts = get_iso_label(range(17))
    else:
        parts = [0]
        parts.extend(list(range(17, 33, 1)))
        parts = get_iso_label(parts)
    
    part_ious = []
    for part in parts:
        I = np.sum(np.logical_and(pred_np == part, seg_np == part))
        U = np.sum(np.logical_or(pred_np == part, seg_np == part))
        if U == 0:
            iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
        else:
            iou = I / float(U)

        part_ious.append(iou)
    shape_ious = np.mean(part_ious)  # part IoU averaged

    return shape_ious
