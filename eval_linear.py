# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util import AverageMeter, learning_rate_decay, load_model, Logger

parser = argparse.ArgumentParser(description="""Train linear classifier on top
                                 of frozen convolutional layers of an AlexNet.""")

parser.add_argument('--data', type=str, help='path to dataset')
parser.add_argument('--model', type=str, help='path to model')
parser.add_argument('--conv', type=int, choices=[1, 2, 3, 4, 5],
                    help='on top of which convolutional layer train logistic regression', default=5)
parser.add_argument('--tencrops', action='store_true',
                    help='validation accuracy averaged over 10 crops')
parser.add_argument('--exp', type=str, default='', help='exp folder')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', type=int, default=90, help='number of total epochs to run (default: 90)')
parser.add_argument('--batch_size', default=10, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--weight_decay', '--wd', default=-4, type=float,
                    help='weight decay pow (default: -4)')
parser.add_argument('--seed', type=int, default=31, help='random seed')
parser.add_argument('--verbose', action='store_true', help='chatty')
parser.add_argument('--no_freeze', action='store_true', help='freeze the model')
parser.add_argument('--rand_weights', action='store_true', help='use random weights')



def main():
    global args
    args = parser.parse_args()
    
    #fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    best_prec1 = 0

    # load model
    model = load_model(args.model, loadWeights=not args.rand_weights)
    model.cuda()
    cudnn.benchmark = True

    # freeze weights
    if (not args.no_freeze):
        for params in model.parameters():
            params.requires_grad = False
        model.eval()
        print("Freezing conv weights")
    else:
        print("fine tuning conv weights")


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # data loading code
    traindir = os.path.join(args.data, 'training')
    valdir = os.path.join(args.data, 'testing')

    normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])

    transforms_train = [transforms.Grayscale(), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]
    transforms_val= [transforms.Grayscale(), transforms.ToTensor(), normalize]

    train_dataset = datasets.ImageFolder(
        traindir,
        transform=transforms.Compose(transforms_train)
    )

    val_dataset = datasets.ImageFolder(
        valdir,
        transform=transforms.Compose(transforms_val)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=int(args.batch_size/2),
                                             shuffle=False,
                                             num_workers=args.workers)

    # logistic regression
    reglog = RegLog(args.conv, len(train_dataset.classes)).cuda()

    # train on all params unless i say so!
    params = list(model.parameters()) + list(reglog.parameters())

    optimizer = torch.optim.SGD(
        params,
        args.lr,
        momentum=args.momentum,
        weight_decay=10**args.weight_decay
    )

    # create logs
    exp_log = os.path.join(args.exp, 'log')
    if not os.path.isdir(exp_log):
        os.makedirs(exp_log)

    loss_log = Logger(os.path.join(exp_log, 'loss_log'))
    prec1_log = Logger(os.path.join(exp_log, 'prec1'))
    prec5_log = Logger(os.path.join(exp_log, 'prec5'))

    for epoch in range(args.epochs):
        end = time.time()

        # train for one epoch
        train(train_loader, model, reglog, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1, prec5, loss = validate(val_loader, model, reglog, criterion, epoch)

        loss_log.log(loss)
        prec1_log.log(prec1)
        prec5_log.log(prec5)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best:
            filename = 'model_best.pth.tar'
        else:
            filename = 'checkpoint.pth.tar'
        torch.save({
            'epoch': epoch + 1,
            'arch': 'alexnet',
            'state_dict': model.state_dict(),
            'prec5': prec5,
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, os.path.join(args.exp, filename))


class RegLog(nn.Module):
    """Creates logistic regression on top of frozen features"""
    def __init__(self, conv, num_labels):
        super(RegLog, self).__init__()
        self.conv = conv
        if conv==1:
            self.av_pool = nn.AvgPool2d(6, stride=6, padding=3)
            s = 9600
        elif conv==2:
            self.av_pool = nn.AvgPool2d(4, stride=4, padding=0)
            s = 9216
        elif conv==3:
            self.av_pool = nn.AvgPool2d(3, stride=3, padding=1)
            s = 3200
        elif conv==4:
            self.av_pool = nn.AvgPool2d(3, stride=3, padding=1)
            s = 3200
        elif conv==5:
            self.av_pool = nn.AvgPool2d(2, stride=2, padding=0)
            s = 2304
        self.linear = nn.Linear(s, num_labels)

    def forward(self, x):
        x = self.av_pool(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.linear(x)
        return x


def forward(x, model, conv):
    count = 1
    for i, m in enumerate(model.features.modules()):
        if not isinstance(m, nn.Sequential):
            x = m(x)
        if isinstance(m, nn.ReLU):
            if count == conv:
                return x
            count = count + 1
    return x

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(train_loader, model, reglog, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        #adjust learning rate
        learning_rate_decay(optimizer, len(train_loader) * epoch + i, args.lr)

        target = target.cuda()
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target)

        # compute output
        output = forward(input_var, model, reglog.conv)

        output = reglog(output)
        loss = criterion(output, target_var)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if args.verbose and i  == len(train_loader) - 1:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        #           'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
        #           .format(epoch, i + 1, len(train_loader), batch_time=batch_time,
        #            data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, reglog, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    data_time = AverageMeter()

    softmax = nn.Softmax(dim=1).cuda()
    end = time.time()
    for i, (input_tensor, target) in enumerate(val_loader):
        data_time.update(time.time() - end)

        target = target.cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input_tensor.cuda())
            target_var = torch.autograd.Variable(target)

        output = reglog(forward(input_var, model, reglog.conv))

  
        output_central = output
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        top1.update(prec1[0], input_tensor.size(0))
        top5.update(prec5[0], input_tensor.size(0))
        loss = criterion(output_central, target_var)
        losses.update(loss.item(), input_tensor.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and i  == len(val_loader) - 1:
            print('Validation: [{0}]    '
                  'Time: {batch_time.avg:.2f}    '
                  'Data: {data_time.avg:.2f}    '
                  'Loss: {loss.avg:.2f}    '
                  'Prec@1: {top1.avg:.1f}    '
                  'Prec@5: {top5.avg:.1f}'
                  .format(epoch, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg

if __name__ == '__main__':
    main()
