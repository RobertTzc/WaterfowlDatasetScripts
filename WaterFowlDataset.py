#Base format of Waterfowl dataset torch loader, all the future implementation should based on this loader or evolve from it.
from logging import root
from anno_util import readTxt
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
        image_name = self.image_dict.keys()[index]
        if ('classification_name' in self.image_dict[image_name]):
            anno_name = self.image_dict[image_name]['classification_name']
        else:
            anno_name = self.image_dict[image_name]['annotation_name']
        image_dir  = self.root_dir+'/'+image_name
        anno_dir = self.root_dir+'/'+anno_name
        image = cv2.cvtColor(cv2.imread(image_dir),cv2.COLOR_BGR2RGB)
        anno_data = readTxt(anno_dir)
        anno_data['altitude'] = self.image_dict['image_name']['altitude']
        anno_data['bbox'] = torch.from_numpy(np.asarray(bbox)).float()
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