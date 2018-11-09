#coding=utf-8
#Writed by zhao

import sys
import os

lr = 0.01
weight_decay = 5e-4
momentum = 0.9
batch_size = 32
lr_steps = [4000, 6000, 8000]
max_iter = 12000
classes_names = {'Spacer_4':1, 'Spacer_6':2, 'Insulator':3, 'Vibration_Damper':4}
cls_num = 5
save_dir = './output/module/'
