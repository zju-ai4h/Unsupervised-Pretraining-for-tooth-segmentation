import os
import os.path as osp
import logging
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from timer import Timer, AverageMeter
from criterion import NCESoftmaxLoss
from torch.serialization import default_restore_location
from model import DGCNN
import torch.distributed as dist

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def load_state(model, weights, lenient_weight_loading=False):
    if get_world_size() > 1:
        _model = model.module
    else:
        _model = model

    if lenient_weight_loading:
        model_state = _model.state_dict()
        filtered_weights = {
            k: v for k, v in weights.items() if k in model_state and v.size() == model_state[k].size()
        }
        logging.info("Load weights:" + ', '.join(filtered_weights.keys()))
        weights = model_state
        weights.update(filtered_weights)

    _model.load_state_dict(weights, strict=True)


class ContrastiveLossTrainer:
    def __init__(
            self,
            args,
            data_loader):

        torch.manual_seed(args.seed)
        if args.cuda:
            logging.info(
                'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(
                    torch.cuda.device_count()) + ' devices')
            torch.cuda.manual_seed(args.seed)
        else:
            logging.info('Using CPU')

        device = torch.device("cuda" if args.cuda else "cpu")
        model = DGCNN(args).to(device)
        model = nn.DataParallel(model)
        logging.info("Let's use " + str(torch.cuda.device_count()) + " GPUs!")

        self.args = args
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr, 
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay)

        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.exp_gamma)
        self.curr_iter = 0
        self.batch_size = args.batch_size
        self.data_loader = data_loader

        # ---------------- optional: resume checkpoint by given path ----------------------
        if args.weight:
            logging.info('===> Loading weights: ' + args.weight)
            state = torch.load(args.weight, map_location=lambda s, l: default_restore_location(s, 'cpu'))
            self.curr_iter = state['curr_iter']
            self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])
            load_state(model, state['state_dict'], args.lenient_weight_loading)

        # ---------------- default: resume checkpoint in current folder ----------------------
        checkpoint_fn = 'weights/weights.pth'
        if osp.isfile(checkpoint_fn):
            logging.info("=> loading checkpoint '{}'".format(checkpoint_fn))
            state = torch.load(checkpoint_fn, map_location=lambda s, l: default_restore_location(s, 'cpu'))
            self.curr_iter = state['curr_iter']
            load_state(model, state['state_dict'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])
        else:
            logging.info("=> no checkpoint found at '{}'".format(checkpoint_fn))

        self.writer = SummaryWriter(logdir='logs')
        if not os.path.exists('weights'):
            os.makedirs('weights', mode=0o755)

    def _save_checkpoint(self, curr_iter, filename='checkpoint'):
        _model = self.model
        state = {
            'curr_iter': curr_iter,
            'state_dict': _model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        filepath = os.path.join('weights', f'{filename}.pth')
        logging.info("Saving checkpoint: {} ...".format(filepath))
        torch.save(state, filepath)
        if os.path.exists('weights/weights.pth'):
            os.remove('weights/weights.pth')
        os.system('ln -s {}.pth weights/weights.pth'.format(filename))


class PointNCELossTrainer(ContrastiveLossTrainer):

    def __init__(
            self,
            args,
            data_loader):
        ContrastiveLossTrainer.__init__(self, args, data_loader)

        self.T = args.nceT
        self.npos = args.npos
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.lr_update_freq = args.lr_update_freq
        
    def compute_loss(self, q, k):
        npos = q.shape[0]
        logits = torch.mm(q, k.transpose(1, 0))  # npos by npos
        labels = torch.arange(npos).cuda().long()
        out = torch.div(logits, self.T)
        out = out.squeeze().contiguous()
        criterion = NCESoftmaxLoss().cuda()
        loss = criterion(out, labels)
        return loss

    def train(self):
        curr_iter = self.curr_iter
        data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()

        for epoch in range(self.epochs):
            for index, data in enumerate(self.data_loader):
                curr_iter += 1
                timers = [data_meter, data_timer, total_timer]
                data_meter, data_timer, total_timer = timers

                self.optimizer.zero_grad()
                batch_loss = 0
                loss = 0
                data_time = 0
                total_timer.tic()

                data_timer.tic()

                data_time += data_timer.toc(average=False)

                feats0 = data['sinput0_F'].to(self.device)
                feats1 = data['sinput1_F'].to(self.device)
               
                feats0 = feats0.permute(0, 2, 1)
                feats1 = feats1.permute(0, 2, 1)

                F0 = self.model(feats0)
                F1 = self.model(feats1)
                F0 = F0.permute(0, 2, 1).contiguous()
                F1 = F1.permute(0, 2, 1).contiguous()

                pos_pairs = data['correspondences']
                for i in range(self.batch_size):
                    pairs = pos_pairs[i].to(self.device)
                    
                    q_unique, count = pairs[:, 0].unique(return_counts=True)
                    uniform = torch.distributions.Uniform(0, 1).sample([len(count)]).to(self.device)
                    off = torch.floor(uniform * count).long()
                    cums = torch.cat([torch.tensor([0], device=self.device), torch.cumsum(count, dim=0)[0:-1]],
                                     dim=0)
                    k_sel = pairs[:, 1][off + cums]
                    if self.npos < q_unique.shape[0]:
                        sampled_inds = np.random.choice(q_unique.shape[0], self.npos, replace=False)
                        q_unique = q_unique[sampled_inds]
                        k_sel = k_sel[sampled_inds]
                    q = F0[i][q_unique.long()]
                    k = F1[i][k_sel.long()]

                    loss += self.compute_loss(q, k)
                loss.backward()
                result = {"loss": loss}
                batch_loss += result["loss"].item()
                self.optimizer.step()
                total_timer.toc()
                data_meter.update(data_time)

                if curr_iter % self.lr_update_freq == 0:
                    lr = self.scheduler.get_last_lr()
                    self.scheduler.step()
                    logging.info(f" Iter: {curr_iter}, LR: {lr}")
                self.writer.add_scalar('train/loss', loss, curr_iter)
                logging.info(
                    "Train Epoch: {:.3f} [{}/{}], Current Loss: {:.3e}"
                    .format(epoch, curr_iter,
                            len(self.data_loader), loss) +
                    "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}, LR: {}".format(
                        data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg,
                        self.scheduler.get_last_lr()))
                data_meter.reset()
                total_timer.reset()
            lr = self.scheduler.get_last_lr()
            logging.info(f" Epoch: {epoch}, LR: {lr}")
            self._save_checkpoint(curr_iter, 'checkpoint_' + str(curr_iter))
