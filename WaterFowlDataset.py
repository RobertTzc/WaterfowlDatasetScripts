import torch
import matplotlib.pyplot as plt
import glob
import numpy as np
import cv2
import torch.utils.data as data
import random
from transform import *
from encoder import DataEncoder
class ListDataset(data.Dataset):
    def __init__(self,txt_dir,cust_transform,transform,input_size,isHeight = False):
        with open(txt_dir,'r') as f:
            self.txt_data = f.readlines()
        self.image_list = [i.split(' ')[0] for i in self.txt_data]
        self.anno_list = [i.split(' ')[1] for i in self.txt_data]
        if (isHeight):
            self.height_list = [int(float(i.split(' ')[2].replace('REF',''))) for i in self.txt_data]
        else:
            self.height_list = []
        self.cust_transform = cust_transform
        self.transform = transform
        self.input_size = input_size
        self.encoder = DataEncoder()
        assert len(self.image_list) == len(self.anno_list)
    def __getitem__(self,index):
        image_dir = self.image_list[index]
        anno_dir  = self.anno_list[index]
        image = Image.open(image_dir).convert("RGB")
        if (anno_dir==''):
            bbox = np.asarray([[]])
        else:
            
            with open(anno_dir,'r') as f:
                anno_data = f.readlines()
            if (anno_data==[]):
                bbox = np.asarray([[]])
            else:
                bbox = []
                for line in anno_data:
                    line = line.replace('\n','').split(',')
                    box = [int(i) for i in line]
                    bbox.append(box)
            bbox = np.asarray(bbox)
        bbox=torch.from_numpy(bbox).float()
        if (self.height_list!=[] and image.size[0]!=512):
            height = self.height_list[index]
            size = int(512.*GSD_calculation(height,'Pro2')/GSD_calculation(90.0,'Pro2'))
            image,bbox = random_crop(image,bbox,size)
            image,bbox = resize(image,bbox,512)
        for trans in self.cust_transform:
            image,bbox =trans(image,bbox)
        image = self.transform(image)
        labels = np.ones(len(bbox))
        labels = torch.from_numpy(np.asarray(labels)).long()
        return image,bbox,labels
    def __len__(self):
        return len(self.anno_list)