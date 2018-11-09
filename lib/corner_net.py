#coding = utf-8
#Writed by zhao

import os
import torch
import torch.nn as nn


class corner_net_with_1_hourglass(nn.Module):
    def __init__(self, num_classes, match = None):
        super(corner_net_with_1_hourglass, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 7, stride=2, padding=0),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU()
                                   )
        self.batch_normal = nn.BatchNorm2d(64)
        self.residual_a = nn.Sequential(nn.Conv2d(64, 32, 1, stride=1, padding=0),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(),
                                        nn.Conv2d(32, 32, 3, stride=1, padding=0),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(32),
                                        nn.Conv2d(32, 128, 1, stride=1, padding=0),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU()
                                        )
        self.pool1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.residual_branch1_a = nn.Sequential(nn.Conv2d(128, 64, 1, stride=1, padding=0),
                                                nn.BatchNorm2d(64),
                                                nn.ReLU(),
                                                nn.Conv2d(64, 64, 3, stride=1, padding=0),
                                                nn.BatchNorm2d(64),
                                                nn.ReLU(),
                                                nn.Conv2d(64, 128, 1, stride=1, padding=0),
                                                nn.BatchNorm2d(128),
                                                nn.ReLU()
                                             )
        self.residual_branch1_b = nn.Sequential(nn.Conv2d(128, 64, 1, stride=1, padding=0),
                                                nn.BatchNorm2d(64),
                                                nn.ReLU(),
                                                nn.Conv2d(64, 64, 3, stride=1, padding=0),
                                                nn.BatchNorm2d(64),
                                                nn.ReLU(),
                                                nn.Conv2d(64, 256, 1, stride=1, padding=0),
                                                nn.BatchNorm2d(256),
                                                nn.ReLU()
                                                )
        self.up_channal = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0)

        self.hour_glass_branch_1_a = nn.Sequential(nn.Conv2d(256, 256, 3, stride=1, padding=1),
                                                   nn.BatchNorm2d(256),
                                                   nn.ReLU(),
                                                   nn.Conv2d(256, 256, 3, stride=1, padding=1),
                                                   nn.BatchNorm2d(256),
                                                   nn.ReLU(),
                                                   nn.Conv2d(256, 512, 3, stride=1, padding=1),
                                                   )
        self.hour_glass_branch_1_b = nn.Sequential(nn.Conv2d(256, 256, 3, stride=1, padding=1),
                                                   nn.BatchNorm2d(256),
                                                   nn.ReLU(),
                                                   nn.Conv2d(256, 256, 3, stride=1, padding=1),
                                                   nn.BatchNorm2d(256),
                                                   nn.ReLU(),
                                                   nn.Conv2d(256, 256, 3, stride=1, padding=1),
                                                   nn.BatchNorm2d(256),
                                                   nn.ReLU(),
                                                   nn.Conv2d(256, 512, 3, stride=1, padding=1),
                                                   nn.BatchNorm2d(512),
                                                   nn.ReLU(),
                                                   nn.Conv2d(512, 512, 3, stride=1, padding=1),
                                                   )
        self.up_sample = nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1)

        self.fc_layer = nn.Sequential(nn.Linear(512, 512),
                                      nn.Linear(512, 256)
                                      )

        self.conv2 = nn.Conv2d(256, num_classes, 1, stride=1,)


        #####################corner pooling TL#####################
        self.conv3_tl_a = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(),
                                        )
        self.conv3_tl_b = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(),
                                        )
        self.conv3_tl_c = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                                        nn.BatchNorm2d(512),
                                        )
        #self.corner_tl_a = sss

        self.conv4_tl = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(128),
                                   )
        self.relu1_tl = nn.ReLU()
        self.conv5_tl = nn.Sequential(nn.Conv2d(128, num_classes, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(num_classes),
                                      nn.ReLU(),
                                      )
        self.conv6_tl_a = nn.Sequential(nn.Conv2d(128, num_classes, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(num_classes),
                                        )
        self.conv6_tl_b = nn.Sequential(nn.Conv2d(128, num_classes, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(num_classes),
                                        )
        self.conv6_tl_c = nn.Sequential(nn.Conv2d(128, num_classes, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(num_classes),
                                        )
        self.conv7_tl_a = nn.Sequential(nn.Conv2d(num_classes, num_classes, kernel_size=1, stride=1, padding=0),
                                        )
        self.conv7_tl_b = nn.Sequential(nn.Conv2d(num_classes, num_classes, kernel_size=1, stride=1, padding=0),
                                        )
        self.conv7_tl_c = nn.Sequential(nn.Conv2d(num_classes, num_classes, kernel_size=1, stride=1, padding=0),
                                        )

        #####################corner pooling BD#####################
        self.conv3_bd_a = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(),
                                        )
        self.conv3_bd_b = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(),
                                        )
        self.conv3_bd_c = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                                        nn.BatchNorm2d(512),
                                        )
        #self.corner_tl_a = sss

        self.conv4_bd = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(128),
                                      )
        self.relu1_bd = nn.ReLU()
        self.conv5_bd = nn.Sequential(nn.Conv2d(128, num_classes, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(num_classes),
                                      nn.ReLU(),
                                      )
        self.conv6_bd_a = nn.Sequential(nn.Conv2d(128, num_classes, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(num_classes),
                                        )
        self.conv6_bd_b = nn.Sequential(nn.Conv2d(128, num_classes, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(num_classes),
                                        )
        self.conv6_bd_c = nn.Sequential(nn.Conv2d(128, num_classes, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(num_classes),
                                        )
        self.conv7_bd_a = nn.Sequential(nn.Conv2d(num_classes, num_classes, kernel_size=1, stride=1, padding=0),
                                        )
        self.conv7_bd_b = nn.Sequential(nn.Conv2d(num_classes, num_classes, kernel_size=1, stride=1, padding=0),
                                        )
        self.conv7_bd_c = nn.Sequential(nn.Conv2d(num_classes, num_classes, kernel_size=1, stride=1, padding=0),
                                        )


    def corner_pooling(self, x):
        #finish corner pooling, there is no parametres in this layer
        return x


    def forward(self, x):
        x = self.conv1(x)

        x = self.batch_normal(x) #batch normalization

        residual = self.up_channal(x, in_channels=64, out_channels=128) #residual 64-128
        x = self.residual_a(x)
        x = x + residual

        x = self.pool1(x) #/2

        residual = x
        x = self.residual_branch1_a(x) #residual 128-128
        x = x + residual

        residual = x
        x = self.residual_branch1_a(x) #residual 128-128
        x = x + residual

        residual = self.up_channal(x)
        x = self.residual_branch1_b(x)  # residual 128-256
        x = x + residual

        residual = self.hour_glass_branch_1_a(x) # hourglass1
        x = self.residual_branch1_b(x)
        backbone_out = x + residual

        # backbone_out = self.fc_layer(x) #no 2 fc layers
        # backbone_out = self.conv2(x) #no last conv

        #####################corner pooling && predict tl#####################
        x_tl_1 = self.conv3_tl_a(backbone_out)
        x_tl_2 = self.conv3_tl_b(backbone_out)
        x_tl_3 = self.conv3_tl_c(backbone_out)

        x_tl_1 = self.corner_pooling(x_tl_1)
        x_tl_2 = self.corner_pooling(x_tl_2)

        x = x_tl_1 + x_tl_2

        x = self.conv4_tl(x)

        x = x + x_tl_3

        x = self.relu1_tl(x)

        x = self.conv5_tl(x)

        x_tl_1 = self.conv6_tl_a(x)
        x_tl_2 = self.conv6_tl_b(x)
        x_tl_3 = self.conv6_tl_c(x)

        heatmaps_tl = self.conv7_tl_a(x_tl_1)
        embeddings_tl = self.conv7_tl_b(x_tl_2)
        offsets_tl = self.conv7_tl_c(x_tl_3)

        #####################corner pooling && predict bd#####################
        x_tl_1 = self.conv3_bd_a(backbone_out)
        x_tl_2 = self.conv3_bd_b(backbone_out)
        x_tl_3 = self.conv3_bd_c(backbone_out)

        x_tl_1 = self.corner_pooling(x_tl_1)
        x_tl_2 = self.corner_pooling(x_tl_2)

        x = x_tl_1 + x_tl_2

        x = self.conv4_bd(x)

        x = x + x_tl_3

        x = self.relu1_bd(x)

        x = self.conv5_bd(x)

        x_tl_1 = self.conv6_bd_a(x)
        x_tl_2 = self.conv6_bd_b(x)
        x_tl_3 = self.conv6_bd_c(x)

        heatmaps_bd = self.conv7_bd_a(x_tl_1)
        embeddings_bd = self.conv7_bd_b(x_tl_2)
        offsets_bd = self.conv7_bd_c(x_tl_3)

        return [heatmaps_tl, embeddings_tl, offsets_tl, heatmaps_bd, embeddings_bd, offsets_bd]


    def save_modle(self, save_dir, epoch):
        cache_file = save_dir + 'corner_net_epoch_{}.pth'.format(epoch)
        print("saving model to {}".format(cache_file))
        with open(cache_file, "wb") as f:
            params = self.model.state_dict()
            torch.save(params, f)

