import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F

from datasets import SagittalCT
from common.model import OrdinalModel
from common.loss import Condor_loss
from common.lr_scheduler import CosineAnnealingWarmUpRestarts
from utils.metric import ordinal_accuracy
from utils.meter import AverageMeter, ProgressMeter
from utils.logger import CompleteLogger
from utils.util import prepare_device, prob_to_label, levels_from_labelbatch
from sklearn.metrics import classification_report, roc_auc_score

from typing import Any, Optional


target_names = ["Normal", "Osteopenia", "Osteoporosis"]


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    train_transform = T.Compose([
        T.Resize((448, 224)),
        T.RandomHorizontalFlip(0.5),
        T.ToTensor(),
        normalize
    ])
    test_transform = T.Compose([
        T.Resize((448, 224)),
        T.ToTensor(),
        normalize
    ])

    train_dataset = SagittalCT(data_dir=args.data_dir, w_min=-450, w_max=1000,
                             train=True, transforms=train_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers)
    val_dataset = SagittalCT(data_dir=args.data_dir, w_min=-450, w_max=1000,
                             train=False, transforms=test_transform)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers)

    device, device_ids = prepare_device(args.n_gpu)
    model = OrdinalModel(args.num_classes)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=1,
                                                 eta_max=0.001, T_up=4, gamma=0.1)

    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        model.load_state_dict(checkpoint)

    if args.phase == 'test':
        acc = validate(model, val_loader, criterion, device, args, True)
        print(acc)
        return

    # start training
    best_acc = 0.
    for epoch in range(args.epochs):
        # train for one epoch
        train(model, train_loader, criterion, optimizer,
              lr_scheduler, epoch, device, args)

        # evaluate on validation set
        acc = validate(model, val_loader, criterion, device, args)

        # remember best acc and save checkpoint
        torch.save(model.state_dict(), logger.get_checkpoint_path('latest'))
        if acc > best_acc:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc = max(acc, best_acc)

    print("best_accuracy = {:3.1f}".format(best_acc))

    # evaluate on test set
    model.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc = validate(model, val_loader, criterion, device, args)
    print("test_accuracy = {:3.1f}".format(acc))

    logger.close()



def train(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: torch.optim, 
         lr_scheduler: torch.optim.lr_scheduler, epoch: int, device: Any, args: argparse.Namespace) -> None:
    """Train model."""
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    accs = AverageMeter('Accuracy', ':3.1f')
    log_step = int(np.sqrt(train_loader.batch_size))
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, accs],
        prefix="Epoch: [{}]".format(epoch)
    )

    model.train()
    end = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(data)

        levels = levels_from_labelbatch(target, args.num_classes)
        levels = levels.to(device)
        loss = criterion(output, levels)

        acc = ordinal_accuracy(output, target)

        losses.update(loss.item(), data.size(0))
        accs.update(acc, data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % log_step == 0:
            progress.display(batch_idx)


def validate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module,
             device: Any, args: argparse.Namespace, use_metric: Optional[bool]=False) -> float():
    """Validate model."""
    batch_time = AverageMeter('Time', ':6.1f')
    losses = AverageMeter('Loss', ':.4e')
    accs = AverageMeter('Accuracy', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, accs],
                             prefix='Test: ')

    model.eval()

    if use_metric:
        outputs = []
        targets = []
    else:
        outputs = None
        targets = None
    with torch.no_grad():
        end = time.time()
        for i, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)

            if outputs is not None:
                outputs.append(output.sigmoid()) 
                targets.append(target)

            levels = levels_from_labelbatch(target, args.num_classes)
            levels = levels.to(device)
            loss = criterion(output, levels)
            acc = ordinal_accuracy(output, target)

            losses.update(loss.item(), data.size(0))
            accs.update(acc, data.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print(' *Acc {accs.avg:.4f}'.format(accs=accs))

        if use_metric:
            outputs = torch.cat(outputs, dim=0)
            targets = torch.cat(targets, dim=0)
            preds = prob_to_label(outputs)         
            
            outputs = outputs.cpu().numpy()
            targets = targets.cpu().numpy()
            preds = preds.cpu().numpy()

            print(classification_report(targets, preds, target_names=target_names, digits=4))
    
    return accs.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diagnoses of osteoporosis in Sagittal CT')
    # dataset parameters
    parser.add_argument('data_dir', metavar='DIR',
                        help='root path of dataset')
    # gpu parameters
    parser.add_argument('--n_gpu', type=int, default=0, metavar='N',
                        help='number of gpus (default: 0)')
    # training parameters
    parser.add_argument('--num_classes', default=3, type=int, metavar='N',
                        help='number of class in dataset')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay',default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=60, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--log", type=str, default='CE',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)