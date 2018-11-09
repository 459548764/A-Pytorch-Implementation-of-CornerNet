#coding=utf-8
#Writed by zhao

import os
import io
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.optim as optim
import numpy as np
import argparse
import lib.config as cfg
from lib.voc import *
from lib.corner_net import corner_net_with_1_hourglass as corner_net
import pandas as pd
from torch.utils.data.sampler import Sampler

parser = argparse.ArgumentParser(description='argumentation of the moxing net～')
# train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--input_file', default='/media/hp208/4t/zhaoxingjie/data/2007_train.txt',
                    help='specify the .txt file of the trainval')
parser.add_argument('--pretrained_dir', default='./model/pretrain.pth')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=1, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='/media/hp208/4t/zhaoxingjie/ssd.pytorch/weight/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


def train_net(num_epochs, loc_loss=None):
    for i in range(num_epochs):
        batch_iterator = iter(train_loader)
        images, targets = next(batch_iterator)
        images = images.type(torch.FloatTensor)
        # targets = targets.type(torch.FloatTensor)
        if args.cuda:
            images = Variable(images.cuda())
            # targets = [Variable(targets.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            # targets = [Variable(ann, volatile=True) for ann in targets]
        out = net(images)
        optimizer.zero_grad()
        loss_l, loss_c = loss_func(out, targets)  # loss_func()
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        loc_loss += loss_l.data[0]
        # c_loss += loss_c.data[0]

        # best_acc = 0.0
        # for epoch in range(num_epochs):
        #     net.train()
        #     train_acc = 0.0
        #     train_loss = 0.0
        #     for i, (images, labels) in enumerate(train_loader):
        #         if args.cuda:
        #             images = Variable(images.cuda())
        #             labels = Variable(labels.cuda())
        #             optimizer.zero_grad()
        #             output = net(images)
        #             loss = loss_func(output, labels)
        #             loss.backward()
        #             optimizer.step()
        #             train_loss += loss.cpu().data[0] * images.size(0)
        #             _, prediction = torch.max(output.data, 1)
        #             train_acc += torch.sum(prediction==labels.data)
        #
        #             train_acc = train_acc / 1000
        #
        #     print("Epoch {}, Train Accuracy: {} , TrainLoss: {} "
        #           .format(epoch, train_acc, train_loss))
        if i%2 == 0:
            net.save_modle(cfg.save_dir, i)


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

net = corner_net(num_classes=6)
print net

# if args.cuda:
#     net.cuda()

train_set = my_datasets(root_dir='./data/VOCdevkit/VOC2007/', target_transform=None)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.batch_size,
                                           shuffle=True, num_workers=args.num_workers)

# initilize the tensor holder here.
im_data = torch.Tensor(1)
im_info = torch.Tensor(1)
num_boxes = torch.LongTensor(1)
gt_boxes = torch.Tensor(1)

# ship to cuda
if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

# make variable
im_data = Variable(im_data)
im_info = Variable(im_info)
num_boxes = Variable(num_boxes)
gt_boxes = Variable(gt_boxes)

optimizer = optim.SGD(net.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
loss_func = nn.CrossEntropyLoss()

if __name__ ==  '__main__':
    train_net(30)


