# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import pickle
import time

import faiss
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import clustering
import models
from util import AverageMeter, UnifLabelSampler
from PIL import Image
from imageio import imwrite
import time
import copy
import csv
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                        choices=['alexnet', 'vgg16'], default='alexnet',
                        help='CNN architecture (default: alexnet)')
    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                        default='Kmeans', help='clustering algorithm (default: Kmeans)')
    parser.add_argument('--k', type=int, default=100,
                        help='number of cluster for k-means (default: 10000)')
    parser.add_argument('--lr', default=0.05, type=float,
                        help='learning rate (default: 0.05)')
    parser.add_argument('--wd', default=-5, type=float,
                        help='weight decay pow (default: -5)')
    parser.add_argument('--reassign', type=float, default=1.,
                        help="""how many epochs of training between two consecutive
                        reassignments of clusters (default: 1)""")
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=256, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--verbose', action='store_true', help='chatty')
    return parser.parse_args()



def main(args):


    # create experiment folder
    exp_folder = "exp_k_" + str(args.k) + "_lr_" + str(args.lr) + "_epochs_" + str(args.epochs)
    print("experiment folder: ", exp_folder)
    if os.path.exists(exp_folder):
        print("experiment folder already exists, please delete: ", exp_folder)
        sys.exit(0)
    else:
        os.makedirs(exp_folder)
        os.makedirs(os.path.join(exp_folder, "checkpoints"))
    
    trainLog = os.path.join(exp_folder, "log.csv")
    csvFields = ["epoch", "cluster_loss", "nn_loss", "nmi_t", "nmi_labels"]
    with open(trainLog, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(csvFields)

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # CNN
    if args.verbose:
        print('Architecture: {}'.format(args.arch))
    model = models.__dict__[args.arch]()
    fd = int(model.top_layer.weight.size()[1])
    model.top_layer = None
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    cudnn.benchmark = True

     # train on all params unless i say so!
    params = list(model.parameters())

    # create optimizer
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=10**args.wd,
    )

    # define loss function
    criterion = nn.CrossEntropyLoss().cuda()

    #normalisation parameters for mnist
    normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])

    # define transforms
    tra = [transforms.Grayscale(), transforms.ToTensor(), normalize]

    # load the data
    dataset = datasets.ImageFolder(args.data, transform=transforms.Compose(tra))

    # 
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch,
                                             num_workers=args.workers,
                                             pin_memory=True)

    index_target_mapping = []
    for i in range(len(dataset)):
        _, target = dataset.__getitem__(i)
        index_target_mapping.append(target)

    # clustering algorithm to use
    deepcluster = clustering.__dict__[args.clustering](args.k)

    # previous image assignments
    image_lists_prev = None

    # training convnet with DeepCluster
    for epoch in range(args.start_epoch, args.epochs):
        end = time.time()

        # remove head
        model.top_layer = None
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

        # get the features for the whole dataset
        features = compute_features(dataloader, model, len(dataset))

        # cluster the features
        clustering_loss = deepcluster.cluster(features, verbose=args.verbose)

        # assign pseudo-labels
        train_dataset = clustering.cluster_assign(deepcluster.images_lists, dataset.imgs)

        # uniformly sample per target
        sampler = UnifLabelSampler(int(args.reassign * len(train_dataset)),
                                   deepcluster.images_lists)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch,
            num_workers=args.workers,
            sampler=sampler,
            pin_memory=True,
        )

        # set last fully connected layer
        mlp = list(model.classifier.children())
        mlp.append(nn.ReLU(inplace=True).cuda())
        model.classifier = nn.Sequential(*mlp)
        model.top_layer = nn.Linear(fd, len(deepcluster.images_lists))
        model.top_layer.weight.data.normal_(0, 0.01)
        model.top_layer.bias.data.zero_()
        model.top_layer.cuda()

        # train network with clusters as pseudo-labels
        end = time.time()
        loss = train(train_dataloader, model, criterion, optimizer, epoch)
        
        # compute nmi between previous and current assignments 1 - no reassignment, 0 - full reassignment
        if image_lists_prev is not None:
            nmi = normalized_mutual_info_score(
                clustering.arrange_clustering(deepcluster.images_lists),
                clustering.arrange_clustering(image_lists_prev))
        else:
            nmi = 0
        image_lists_prev = copy.deepcopy(deepcluster.images_lists)


        # compute nmi between true labels and clusters
        labels  = []
        for imageList in deepcluster.images_lists:
            for i in imageList:
                labels.append(index_target_mapping[i])

        nmi_labels = normalized_mutual_info_score(
            clustering.arrange_clustering(deepcluster.images_lists),
            labels)

        if args.verbose:
            print("Epoch: {}  Time: {:3.2f}  cluster loss: {:6.2f}  nn loss: {:2.3f} nmi t-1/t: {:1.3f} nmi labels: {:1.3f}"
                    .format(epoch, time.time() - end, clustering_loss, loss, nmi, nmi_labels))

        # log  
        with open(trainLog, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, clustering_loss, loss, nmi, nmi_labels])



        if epoch % 10 == 0:
            # save running checkpoint
            torch.save({'epoch': epoch,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict()},
                    os.path.join(os.path.join(exp_folder, "checkpoints"), 'checkpoint_' + str(epoch) + '.pth.tar'))






def train(loader, model, crit, opt, epoch):
    """Training of the CNN.
        Args:
            loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): CNN
            crit (torch.nn): loss
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch (int)
    """
    losses = AverageMeter()

    # switch to train mode
    model.train()

    # create an optimizer for the last fc layer
    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(),
        lr=args.lr,
        weight_decay=10**args.wd,
    )


    for input_tensor, target in loader:
        target = target.cuda()
        input_var = torch.autograd.Variable(input_tensor.cuda())
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = crit(output, target_var)

        # record loss
        losses.update(loss.item(), input_tensor.size(0))

        # compute gradient and do SGD step
        opt.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        opt.step()
        optimizer_tl.step()

    return losses.avg


def compute_features(dataloader, model, N):

    # dont compute gradients
    model.eval()

    # discard the label information in the dataloader
    for i, (input_tensor, _) in enumerate(dataloader):
        with torch.no_grad():
            input_var = torch.autograd.Variable(input_tensor.cuda())
        aux = model(input_var).data.cpu().numpy()
        
        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * args.batch: (i + 1) * args.batch] = aux
        else:
            # special treatment for final batch
            features[i * args.batch:] = aux

    return features


if __name__ == '__main__':
    args = parse_args()
    main(args)
