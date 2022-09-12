#Base format of Waterfowl dataset torch loader, all the future implementation should based on this loader or evolve from it.
import sys
sys.path.append('/home/zt253/data/WaterfowlDataset/WaterfowlDatasetScripts/Split_data')
from logging import root
from anno_util import readTxt
from cust_transform import *
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.utils.data as data
import pandas as pd
from collections import defaultdict


class WaterFowlDataset(data.Dataset):
    def __init__(self,root_dir,csv_dir,cust_transform,torch_transform,task = 'altitude_split_Robert',phase = 'Train',**kwargs):
        assert task in ['altitude_split_Robert','bbox_split_Robert','category_split_Robert']
        df = pd.read_csv(csv_dir)
        self.image_dict = defaultdict(dict)
        for idx in range(len(df)):
            item = df.iloc[idx]
            self.image_dict[item['image_name']] = dict()
            for keys in item.keys():
                if (keys== 'image_name'):
                    pass
                else:
                    self.image_dict[item['image_name']][keys] = item[keys]
        self.root_dir = root_dir            
        self.task = task
        self.phase = phase
        self.cust_transform = cust_transform
        self.torch_transform = torch_transform
        self.additinal_args = kwargs    
    def __getitem__(self,index):
        image_name = list(self.image_dict.keys())[index]
        if ('classification_name' in self.image_dict[image_name]):
            anno_name = self.image_dict[image_name]['classification_name']
        else:
            anno_name = self.image_dict[image_name]['annotation_name']
        image_dir  = self.root_dir+'/'+image_name
        anno_dir = self.root_dir+'/'+anno_name
        image = cv2.cvtColor(cv2.imread(image_dir),cv2.COLOR_BGR2RGB)
        anno_data = readTxt(anno_dir)
        anno_data['altitude'] = self.image_dict[image_name]['height']
        
        
        image,anno_data = self.cust_transform(image,anno_data)
        anno_data['bbox'] = torch.from_numpy(np.asarray(anno_data['bbox'])).float()
        if (self.torch_transform):
            image = self.torch_transform(image)
        detection_labels = np.ones(len(anno_data['bbox']))
        detection_labels = torch.from_numpy(np.asarray(detection_labels)).long()
        anno_data['detection_labels'] = detection_labels
        return image,anno_data
    def __len__(self):
        return len(self.image_dict)
    
if __name__ == '__main__':
    root_dir = '/home/zt253/data/WaterfowlDataset/Processed/Bird_D'
    csv_dir = '/home/zt253/data/WaterfowlDataset/Processed/Bird_D/image_info.csv'
    cust_transform = cust_sequence([RandomHorizontalFlip,RandomVerticalFlip,random_rotate],0.5)
    torch_transform = None
    task = 'bbox_split_Robert'
    phase = 'Train'
    dataset = WaterFowlDataset(root_dir=root_dir,csv_dir = csv_dir,cust_transform = cust_transform,torch_transform=None,task = task,phase = phase)
    for image,anno in dataset:
        print (image.shape)
        