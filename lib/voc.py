#coding=utf-8
#Writed by zhao

import os
import torch
from torch.utils.data import Dataset
import cv2
import xml.etree.ElementTree as ET
from lib import config as cfg


class my_datasets(Dataset):
    def __init__(self, root_dir, target_transform=None):
        self.root_dir = root_dir
        self.name_list = []
        self.ext = '.JPG' + '\n'
        # with open(txt_file) as f: #when the trainval & test pic in the same folder
        #     name = f.readlines()
        # for line in open(txt_file):
        #     self.name_list.append((root_dir + 'JPEGImages/' + line.rstrip('\n') + '.jpg'))
        self.img_dir = root_dir + 'JPEGImages/'
        self.target_transform = target_transform
        self.img_name = os.listdir(self.img_dir)
        self.name_list = [os.path.join(self.img_dir, x)for x in os.listdir(self.img_dir)]
        self.landmarks_frame = self.img_name

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        image = cv2.imread(self.name_list[idx])
        image = cv2.resize(image, (356, 536)) #1/8 of the width ,and 1/8 of the height
        image = image[:, :, (2, 1, 0)]
        anno_name = os.path.join(self.root_dir, 'Annotations/', self.landmarks_frame[idx]).rstrip('.JPG\n') + '.xml'
        tree = ET.parse(anno_name)
        root = tree.getroot()
        # size = root.find('size')
        # w = int(size.find('width').text)#读图的时候已经有图像的宽高了
        # h = int(size.find('height').text)
        gt = []
        # gt.append(image)
        for obj in root.iter('object'):
            cls = obj.find('name').text
            cls = cfg.classes_names[cls]
            gt.append(cls)
            box = obj.find('bndbox')
            [x,  y, w, h] = int(box.find('xmin').text), int(box.find('ymin').text), \
                          int(box.find('xmax').text) - int(box.find('xmin').text), \
                          int(box.find('ymax').text) - int(box.find('ymin').text)
            gt.append([x,y,w,h])
        # gt = { 'labels': cls, 'bbox': [x, y, w, h]}
        # if self.transform:
        #     sample = self.transform(sample)

        return torch.from_numpy(image).permute(2, 0, 1), gt
        # return image, gt