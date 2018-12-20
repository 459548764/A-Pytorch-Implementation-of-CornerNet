#!usr/bin/python
#coding=utf-8
# @Time    : 2018-10-14 10:09
# @Author  : zhaowujie
# @Email   : 18829350080@163.com
# @Software: PyCharm


import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.optim as optim
import numpy as np
import argparse
import test.config.config as cfg

from dataset import mydataset
from modle import moxing_model

'''
参数
'''
parser = argparse.ArgumentParser(description='argumentation of the moxing net～')
# train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--input_file', default='/media/hp208/4t/zhaoxingjie/data/2007_train.txt',
                    help='specify the .txt file of the trainval')
parser.add_argument('--pretrained_dir', default='./model/pretrain.pth')
parser.add_argument('--batch_size', dest='batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('--epoch', default=30, type=int,
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
parser.add_argument('--save_folder', default='./weight/',
                    help='Directory for saving checkpoint models')


def train(num_epochs, loss=None):
    for epoch in range(num_epochs):
        net.train()
        train_acc = 0.0
        train_loss = 0.0
        for batch_i, (_, imgs, targets) in enumerate(train_loader):
            imgs = Variable(imgs.type(Tensor))
            targets = Variable(targets.type(Tensor), requires_grad=False)
            optimizer.zero_grad()
            out = net(imgs)
            loss_l, loss_c = loss_func(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            train_loss += loss.cpu().data[0] * imgs.size(0)
            _, prediction = torch.max(out.data, 1)
            train_acc += torch.sum(prediction == targets.data)
            train_acc = train_acc / 1000
            print 'epoch {}, batch {}, loss_l = {}, loss_c = {}, loss = {}'.format(epoch, batch_i, loss_l, loss_c, loss)
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {} ".format(epoch, train_acc, train_loss))
    torch.save(net.state_dict(), )


if __name__ ==  '__main__':
    args = parser.parse_args()
    weight_dir = args.save_folder + 'final_weights.pth'
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    net = moxing_model(num_classes=cfg.cls_num, batch_size=args.batch_size)
    print (net)

    if args.cuda:
        net = net.cuda()  # put network on gpu
    net.train()

    optimizer = optim.SGD(net.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    loss_func = nn.MSELoss()
    '''
    数据
    '''
    data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([100, 100, 100], [100, 100, 100])])

    train_set = mydataset(root_path='/media/hp208/4t/zhaoxingjie/data/VOCdevkit2007/VOC2007')
    train_size = train_set.__len__()
    # sampler_batch = sampler(train_size, args.batch_size)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers)

    Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

    train(args.epoch)