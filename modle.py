#coding=utf-8
import torch
import torch.nn as nn
import test.config.config as cfg
import numpy as np


class moxing_model(nn.Module):
    def __init__(self, num_classes, batch_size):
        super(moxing_model, self).__init__()#重写方法
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.max_in_per_img = cfg.max_in_per_img
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  #used to overcome overfitting, only in the training phase,
                                # in predict phase ,all nodes were reserved, instead ,use zoom out/in these paras
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, stride=3, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(20, 20), stride=1, padding=0),#need calculate the size of output ？
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 32, 3, stride=3, padding=0),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, self.num_classes, 3, stride=3, padding=0),
            nn.ReLU(inplace=True),
        )
        self.line_layer = nn.Sequential(
            nn.Linear(self.num_classes * self.batch_size * 5 * 5,
                      5 * self.batch_size * self.max_in_per_img),
            nn.Dropout(p=0.5),
        )
        # self.cls_predict_layer = nn.Linear(self.num_classes * self.batch_size * self.max_in_per_img,
        #                                self.num_classes * self.batch_size * self.max_in_per_img) #number of neuron in last layer
        # self.bbox_predict_layer = nn.Linear(self.num_classes * self.batch_size * self.max_in_per_img,
        #                                    4 * self.batch_size * self.max_in_per_img)
        # self.soft_max = nn.Softmax()
        # self.conv6 = nn.Sequential(
        #     nn.Conv2d(256, 128, kernel_size=3, padding=0)
        # )

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)    #8 256  208 208

        cls_conv3 = self.conv3(conv2)    #x.shape (batchsize，channels，x，y) 8  128  50  50
        cls_conv4 = self.conv4(cls_conv3)  #8  32  16  16
        cls_conv5 = self.conv5(cls_conv4)   #8  6  5  5
        feat = cls_conv5.view(cls_conv5.size(0) * cls_conv5.size(1) * cls_conv5.size(2) * cls_conv5.size(3)) #specify the number of lines of matrix, 3d data =>2d matrix
        out = self.line_layer(feat)
        out = out.reshape(self.batch_size, self.max_in_per_img, 5) #8  20  6)
        # x = x.view(256, -1)
        # cls_pred = self.cls_predict_layer(feat)
        # cls_pred = self.soft_max(cls_pred) #softmax in channal dim
        # cls_pred = cls_pred.reshape(self.batch_size, self.max_in_per_img, self.num_classes) #8  20  6
        #
        # filled_labels = np.zeros((self.batch_size, self.max_in_per_img, 1))
        # cls_pred = cls_pred.cpu()
        # cls_pred = cls_pred.numpy()
        #
        # bb_pred = self.bbox_predict_layer(feat)
        # bb_pred = self.soft_max(bb_pred)
        # bb_pred = bb_pred.reshape(self.batch_size, self.max_in_per_img, 4)
        # bb_pred = bb_pred.cpu()
        # bb_pred = bb_pred.numpy()
        #
        # for i in range(self.batch_size):
        #     for j in range(self.max_in_per_img):
        #         #np.max(cls_pred[i][j])
        #         lab = (cls_pred[i][j]).tolist()
        #         cls_id = lab.index(max(lab))
        #         filled_labels[i][j] = cls_id
        #
        # out = np.append(filled_labels, bb_pred, axis=2)  #stack together
        #
        # out = torch.from_numpy(out)
        # out = out.cuda()

        return out
