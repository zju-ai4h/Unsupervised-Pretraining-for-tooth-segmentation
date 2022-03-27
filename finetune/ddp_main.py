from module import train,test
import os
import argparse
import torch
from model import DGCNN
import numpy as np
from torch.utils.data import DataLoader
from util import IOStream, load_state_with_same_shape
from dataset import TeethDataset
import torch.nn as nn


def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name):
        os.makedirs('outputs/'+args.exp_name)
    if not os.path.exists('outputs/'+args.exp_name+'/'+'models'):
        os.makedirs('outputs/'+args.exp_name+'/'+'models')

def main(args):

    _init_()

    io = IOStream('outputs/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')
    model = DGCNN(args)
    if args.pretrain:
        print('===> Loading weights: ' + args.pretrain)
        state = torch.load(args.pretrain)
        if 'state_dict' in state.keys():
           state_key_name = 'state_dict'
        elif 'model_state' in state.keys():
           state_key_name = 'model_state'
        else:
           raise NotImplementedError

        if args.lenient_weight_loading:
            matched_weights = load_state_with_same_shape(model, state[state_key_name])
            model_dict = model.state_dict()
            print(matched_weights.keys())
            model_dict.update(matched_weights)
            model.load_state_dict(model_dict)
        else:
            model.load_state_dict(state['state_dict'])
    device = torch.device("cuda" if args.cuda else "cpu")
    model = model.to(device)
    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    if not args.eval:
        train_loader = DataLoader(TeethDataset(partition='train'), 
                              num_workers=4, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(TeethDataset(partition='val'), 
                            num_workers=4, batch_size=args.val_batch_size, shuffle=True, drop_last=False)
        train(model, train_loader, val_loader, device, args, io)
    else:
        test_loader = DataLoader(TeethDataset(partition='test'), 
                            num_workers=2, batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        test(model, test_loader, device, args, io)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Teeth Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N', help='Name of the experiment')
    
    parser.add_argument('--train_batch_size', type=int, default=4, metavar='batch_size', help='Size of batch)')
    
    parser.add_argument('--val_batch_size', type=int, default=4, metavar='batch_size', help='Size of batch)')
    
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size', help='Size of batch)')
    
    parser.add_argument('--epochs', type=int, default=400, metavar='N', help='number of episode to train ')
    
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001, 0.1 if using sgd)')
    
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    
    parser.add_argument('--no_cuda', type=bool, default=False, help='enables CUDA training')
    
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    
    parser.add_argument('--eval', type=bool,  default=False, help='evaluate the model')

    parser.add_argument('--k', type=int, default=25, metavar='N', help='Num of nearest neighbors to use')
    
    parser.add_argument('--model_path', type=str, default=None, help='path to restore the model [default: None]')
    
    parser.add_argument('--val_stat', type=int, default='1', metavar='N', help='val frequency')
    
    parser.add_argument('--pretrain', type=str, default=None, metavar='N', help='pretrain_root')
    
    parser.add_argument('--lenient_weight_loading', type=bool, default=False, help='fit model')
    
    parser.add_argument('--num_features', type=int, default=15, help='Number of features per point')
    args = parser.parse_args()
    main(args)
