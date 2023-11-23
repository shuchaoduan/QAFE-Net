import argparse
import os
import random
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from einops import rearrange
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataloader.Parkinson_landmarkheatmap import train_data_loader, test_data_loader
from models.slowonly import ResNet3dSlowOnly
from models.ST_Former import GenerateModel
from dataloader.ibmse.balancedMSE import GAILoss, BMCLoss, FocalRLoss
import numpy as np
import datetime

from models.evaluator import Evaluator
from models.vit_decoder_two import decoder_fuser


parser = argparse.ArgumentParser()
parser.add_argument('--class_idx',type=int,default=0,choices=[0, 1, 2, 3, 4], help='class idx in PD-5')
parser.add_argument('--clip_len',type=int,default=80, help='input length')
parser.add_argument('--data_root',type=str,default='/path/PFED5/frames', help='data path')
parser.add_argument('--landmarks_root',type=str,default='/path/PFED5/landmarks_heatmap', help='landmarks path')
parser.add_argument('--avreage_times',type=int,help='sample frames in X times for inference',default=10)
parser.add_argument('--exp_name',type=str,help='path to save tensorboard curve',default='test')
parser.add_argument('--seed',type=int,help='manual seed',default=1)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=1, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--noise_sigma', type=float, default=1.)




args = parser.parse_args()
now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H:%M]-")
output_log_root = './log/' +'{}/'.format(args.exp_name)
if not os.path.exists(output_log_root):
    os.makedirs(output_log_root)
output_ckpt_root = './checkpoint/' +'{}/'.format(args.exp_name)
if not os.path.exists(output_ckpt_root):
    os.makedirs(output_ckpt_root)
log_txt_path = os.path.join(output_log_root, 'class_{}-log.txt'.format(args.class_idx))
log_curve_path = os.path.join(output_log_root, 'class_{}-loss.png'.format(args.class_idx))
checkpoint_path = os.path.join(output_ckpt_root,  'class_{}-model.pth'.format( args.class_idx))
best_checkpoint_path = os.path.join(output_ckpt_root,  'class_{}-model_best.pth'.format(args.class_idx))
model_pretrained_path = './models/FormerDFER-DFEWset1-model_best.pth'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def init_seed(args):
    """
    Set random seed for torch and numpy.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

init_seed(args)

def main():
    # print configuration
    print('=' * 40)
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print('=' * 40)

    best_acc = 0
    best_epoch = 0
    recorder = RecorderMeter(args.epochs)
    print('The training time: ' + now.strftime("%m-%d %H:%M"))

    # create model and load pre_trained parameters
    model = GenerateModel().cuda()
    SlowOnly = ResNet3dSlowOnly().cuda()
    evaluator = Evaluator(output_dim=1, model_type='MLP').cuda()
    decoder = decoder_fuser(dim=512, num_heads=8, num_layers=3, drop_rate=0.).cuda()

    from collections import OrderedDict

    new_state_dict = OrderedDict()
    pre_trained_dict = torch.load(model_pretrained_path)
    for k, v in pre_trained_dict['state_dict'].items():
        name = k[7:]
        new_state_dict[name] = v
    new_state_dict.pop('fc.weight')
    new_state_dict.pop('fc.bias')
    model.load_state_dict(new_state_dict)
    SlowOnly.load_weights() ## (S)

    
    if len(args.gpu.split(',')) > 1:
        model = nn.DataParallel(model)
        evaluator = nn.DataParallel(evaluator)
        SlowOnly = nn.DataParallel(SlowOnly)
        decoder = nn.DataParallel(decoder)

    # define loss function (criterion) and optimizer
    criterion = BMCLoss(init_noise_sigma=args.noise_sigma)

    optimizer = torch.optim.SGD([{'params': model.parameters()},
                                {'params': evaluator.parameters()},
                                 {'params': decoder.parameters()},
                                 {'params': SlowOnly.parameters()}],
                                args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            recorder = checkpoint['recorder']
            best_acc = best_acc.cuda()
            SlowOnly.load_state_dict(checkpoint['SlowOnly'])
            decoder.load_state_dict(checkpoint['decoder'])
            model.load_state_dict(checkpoint['model'])
            evaluator.load_state_dict(checkpoint['evaluator'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True

    # Data loading code
    train_data = train_data_loader(args)
    test_data = test_data_loader(args)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        inf = '********************' + str(epoch) + '********************'
        start_time = time.time()
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']

        with open(log_txt_path, 'a') as f:
            f.write(inf + '\n')
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')

        print(inf)
        print('Current learning rate: ', current_learning_rate)

        # train for one epoch
        train_acc, train_los = train(train_loader, model, evaluator, SlowOnly, decoder, criterion, optimizer, epoch, args)
        # evaluate on validation set
        val_acc, val_los = validate(val_loader, model, evaluator, SlowOnly, decoder, criterion, args)

        scheduler.step()

        # remember best acc and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        if is_best:
            best_epoch = epoch
        save_checkpoint({'epoch': epoch + 1,
                         'model': model.state_dict(),
                         'evaluator': evaluator.state_dict(),
                         'SlowOnly': SlowOnly.state_dict(),
                         'decoder': decoder.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'recorder': recorder}, is_best)

        # print and save log
        epoch_time = time.time() - start_time


        recorder.update(epoch, train_los, val_los)
        recorder.plot_curve(log_curve_path)

        print('The best rho: {:.5f} in epoch {}'.format(best_acc, best_epoch))
        print('An epoch time: {:.1f}s'.format(epoch_time))
        with open(log_txt_path, 'a') as f:
            f.write('The best rho: {:.5f}' + str(best_acc) + 'in {}'.format(best_epoch) + '\n')
            f.write('An epoch time: {:.1f}s' + str(epoch_time) + '\n')


def train(train_loader, model, evaluator, SlowOnly, decoder, criterion, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(len(train_loader),
                             [losses],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    SlowOnly.train()
    decoder.train()
    evaluator.train()
    true_scores = []
    pred_scores = []

    for idx, data in enumerate(train_loader):

        true_scores.extend(data['final_score'].numpy())
        videos = data['video'].cuda()
        heatmaps = data['landmark_heatmap'].cuda()
        b = data['final_score'].unsqueeze_(1).type(torch.FloatTensor).cuda()

        # compute output
        data_pack = torch.cat(
                [videos[:, :, i:i + 16] for i in range(0, args.clip_len, 16)]).cuda()  # 5xN, c, 16, h, w
        outputs_v = model(data_pack).reshape(5, len(videos), 512).transpose(0, 1)  # N, 5, featdim
        heatmap_pack = torch.cat(
                [heatmaps[:, :, i:i + 16] for i in range(0, args.clip_len, 16)]).cuda()  # 5xN, c, 16, h, w
        outputs_l = SlowOnly(heatmap_pack).reshape(5, len(videos), 512).transpose(0, 1) # [b, 5, 512]

        output_lv_map = decoder(outputs_l, outputs_v)  # q, v
        probs = evaluator(output_lv_map)
        probs = probs.mean(1)
        preds=probs
        loss = criterion(preds, b)
        losses.update(loss.item(), videos.size(0))

        pred_scores.extend([i.item() for i in preds])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

       # print loss and accuracy
        if idx % args.print_freq == 0 or idx == len(train_loader)-1:
            progress.display(idx)

    rho_v, p_v = stats.spearmanr(pred_scores, true_scores)
    print('[train] EPOCH: %d,  correlation_v: %.4f,  lr: %.4f'
          % (epoch,  rho_v,  optimizer.param_groups[0]['lr']))

    if epoch == 2 or epoch == args.epochs-1:
        print('pred_v scores', pred_scores)
        print('true_scores', true_scores)

    return rho_v, losses.avg


def validate(val_loader, model, evaluator, SlowOnly, decoder, criterion, args):
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(len(val_loader),
                             [losses],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    evaluator.eval()
    SlowOnly.eval()
    decoder.eval()

    true_scores = []
    pred_scores = []

    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            true_scores.extend(data['final_score'].numpy())
            videos = data['video'].cuda()
            heatmaps = data['landmark_heatmap'].cuda()
            b = data['final_score'].unsqueeze_(1).type(torch.FloatTensor).cuda()

            # compute output
            data_pack = torch.cat(
                        [videos[:, :, i:i + 16] for i in range(0, args.clip_len, 16)]).cuda()  # 5xN, c, 16, h, w
            outputs_v = model(data_pack).reshape(5, len(videos), 512).transpose(0, 1)  # N, 5, featdim
            heatmap_pack = torch.cat(
                        [heatmaps[:, :, i:i + 16] for i in range(0, args.clip_len, 16)]).cuda()  # 5xN, c, 16, h, w
            outputs_l = SlowOnly(heatmap_pack).reshape(5, len(videos), 512).transpose(0, 1) # [b, 5, 512]

            output_lv_map = decoder(outputs_l, outputs_v)  # q, v
            probs = evaluator(output_lv_map)
            probs = probs.mean(1)
            preds = probs
            loss = criterion(preds, b)

            pred_scores.extend([i.item() for i in preds])

            losses.update(loss.item(), videos.size(0))


            if idx % args.print_freq == 0 or idx == len(val_loader)-1:
                progress.display(idx)

        rho_v, p_v = stats.spearmanr(pred_scores, true_scores)

        # TODO: this should also be done with the ProgressMeter
        print('Current Accuracy: {rho_v:.6f}'.format(rho_v=rho_v))
        print('Predicted visual scores: ', pred_scores)
        print('True scores: ', true_scores)
        with open(log_txt_path, 'a') as f:
            f.write('Current Accuracy:  {rho_v:.6f}'.format(rho_v=rho_v) + '\n')
    return rho_v, losses.avg


def save_checkpoint(state, is_best):
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_checkpoint_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        with open(log_txt_path, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)
#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
#         res = []
#         for k in topk:
#             correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]


    def update(self, idx, train_los, val_los):
        self.epoch_losses[idx, 0] = train_los * 50
        self.epoch_losses[idx, 1] = val_los * 50

        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1600, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 1
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_losses_v[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-loss-v-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses_v[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-loss-v-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses_l[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='--', label='train-loss-l-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses_l[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='--', label='valid-loss-l-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    main()
