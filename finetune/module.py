from torch.utils.data import DataLoader
from dataset import TeethDataset
from util import cal_loss, IOStream, load_state_with_same_shape
import sklearn.metrics as metrics
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import numpy as np
from tqdm import tqdm
import torch
import os
import logging

def calculate_sem_IoU(pred_np, seg_np, visual=False):
    I_all = np.zeros(33)
    U_all = np.zeros(33)
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(33):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U

    return I_all / U_all


def train(model, train_loader, val_dataloader, device, args, io):

    #for name,parameters in model.named_parameters():
    #    print(name,':',parameters)
    opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    criterion = cal_loss
    best_test_iou = 0
    best_val = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for index, (data, target, category) in loop:
            loop.set_description(f'Epoch [{epoch}/{args.epochs}]')
            data, target, category = data.to(device), target.to(device), category.to(device).float()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            
            opt.zero_grad()
            
            seg_pred = model(data, category)

            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, 33), target.view(-1, 1).squeeze())
            loss.backward()
            opt.step()
            pred = seg_pred.max(dim=2)[1]  # (batch_size, num_points)

            count += batch_size
            train_loss += loss.item() * batch_size
            seg_np = target.cpu().numpy()  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()  # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))  # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))  # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)


        scheduler.step()
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_ious = calculate_sem_IoU(train_pred_seg, train_true_seg)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (epoch,
                                                                                                  train_loss * 1.0 / count,
                                                                                                  train_acc,
                                                                                                  avg_per_class_acc,
                                                                                                  np.mean(train_ious))
        io.cprint(outstr)
        
        if epoch % args.val_stat == 0:
            with torch.no_grad():
                ####################
                # Test
                ####################
                test_loss = 0.0
                count = 0.0
                model.eval()
                test_true_cls = []
                test_pred_cls = []
                test_true_seg = []
                test_pred_seg = []

                for data, seg, category in val_dataloader:
                    data, seg, category = data.to(device), seg.to(device), category.to(device).float()

                    data = data.permute(0, 2, 1)
                    batch_size = data.size()[0]

                    seg_pred = model(data, category)

                    seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                    loss = criterion(seg_pred.view(-1, 33), seg.view(-1, 1).squeeze())

                    pred = seg_pred.max(dim=2)[1]
                    count += batch_size
                    test_loss += loss.item() * batch_size
                    seg_np = seg.cpu().numpy()
                    pred_np = pred.detach().cpu().numpy()
                    test_true_cls.append(seg_np.reshape(-1))
                    test_pred_cls.append(pred_np.reshape(-1))
                    test_true_seg.append(seg_np)
                    test_pred_seg.append(pred_np)
                test_true_cls = np.concatenate(test_true_cls)
                test_pred_cls = np.concatenate(test_pred_cls)
                test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
                avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
                test_true_seg = np.concatenate(test_true_seg, axis=0)
                test_pred_seg = np.concatenate(test_pred_seg, axis=0)
                test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
                outstr = 'Val %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (epoch,
                                                                                                     test_loss * 1.0 / count,
                                                                                                     test_acc,
                                                                                                     avg_per_class_acc,
                                                                                                     np.mean(test_ious))
                io.cprint(outstr)
                if np.mean(test_ious) >= best_test_iou:
                    best_test_iou = np.mean(test_ious)
                    best_val = epoch
                    torch.save(model.state_dict(), 'outputs/%s/models/checkpoint.pth' % (args.exp_name))
                outstr = 'Best Val %d, best iou: %.6f' % (best_val, best_test_iou)
                io.cprint(outstr)


def test(model, test_loader, device, args, io):
    with torch.no_grad():
        # Try to load models
        model.load_state_dict(torch.load(os.path.join(args.model_root)))
        for name,parameters in model.named_parameters():
          print(name,':',parameters)
        model = model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        for data, seg, category in test_loader:
            data, seg, category = data.to(device), seg.to(device), category.to(device).float()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            seg_pred = model(data, category)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()

            pred = seg_pred.max(dim=2)[1]

            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()

            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)

        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
        outstr = 'Test :: test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (test_acc,
                                                                                 avg_per_class_acc,
                                                                                 np.mean(test_ious))
        io.cprint(outstr)
