#coding = utf-8

import os
import numpy as np
from torch.autograd import Variable
import torch
import torchvision
import torch.nn as nn
from visualization import make_dot

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(1, 6, 3, padding=1),
                                    nn.MaxPool2d(2, 2)
                                    )
        self.layer2 = nn.Sequential(nn.Conv2d(6, 16, 5),
                                    nn.MaxPool2d(2, 2)
                                    )
        self.layer3 = nn.Sequential(nn.Linear(400, 120),
                                    nn.Linear(120, 84),
                                    nn.Linear(84, 10)
                                    )
    def forward(self, *input):
        x=self.layer1(*input)
        x=self.layer2(x)
        x=self.layer3(x)
        return x

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.Conv2d(64, 192, kernel_size=5, padding=2),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.Conv2d(192, 384, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(384, 256, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2)
                                      )
        self.classifier = nn.Sequential(nn.Dropout(),
                                        nn.Linear(256 * 6 * 6, 4069),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(4069, 4069),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(4069, num_classes)
                                        )

    def forward(self, *input):
        x = self.features(*input)
        x = x.view(x.size(0), 256 * 6 * 6) #conv to linear must transform the 3d-data into 2d
        x = self.classifier(x)
        return x

class VGGNet(nn.Module):
    def __init__(self, nclasses):
        super(VGGNet, self).__init__()

        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Conv2d(256, 512, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2)
                                      )
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(4096, nclasses)
                                        )
        # self._initialize_weights()

    def forward(self, *input):
        x = self.features(*input)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train():
    img = []




if __name__ == '__main__':
    net1 = LeNet()
    net2 = AlexNet(num_classes=4)
    net3 = VGGNet(nclasses=5)
    print net1, net2, net3

    x = np.arange(2 * 299 * 299 * 3)
    x = x.reshape(2, 3, 299, 299)
    x = x / float(x.max())
    x = torch.from_numpy(x)
    x = x.float()
    x = Variable(x)

    y = net3(x)
    g = make_dot(y)
    # g.view()
    g.render('here', view=False)

    train()
