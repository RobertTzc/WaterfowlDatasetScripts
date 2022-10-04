# Base format of Waterfowl dataset torch loader, all the future implementation should based on this loader or evolve from it.

from logging import root
from WaterFowlTools.anno_util import readTxt
from WaterFowlTools.cust_transform import *
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.utils.data as data
import pandas as pd
from collections import defaultdict



class WaterFowlDataset(data.Dataset):
    def __init__(self, root_dir, csv_dir, cust_transform, torch_transform, task='altitude_split_Robert', phase='train', **kwargs):
        assert task in ['altitude_split_Robert', 'bbox_split_Robert',
                        'category_split_Robert'], "Current only support splits from \['altitude_split_Robert','bbox_split_Robert','category_split_Robert'\]"
        assert phase in [
            'train', 'test'], "Phase only takes either 'train' or 'test'"
        self.image_dict = defaultdict(dict)
        df = pd.read_csv(csv_dir)
        assert task in df.columns, "{} is not found in current csv files: {}".format(
            task, csv_dir)
        for idx in range(len(df)):
            item = df.iloc[idx]
            if (item[task] == phase):
                self.image_dict[item['image_name']] = dict()
                for keys in item.keys():
                    if (keys == 'image_name'):
                        pass
                    else:
                        self.image_dict[item['image_name']][keys] = item[keys]
        self.root_dir = root_dir
        self.task = task
        self.phase = phase
        self.cust_transform = cust_transform
        self.torch_transform = torch_transform
        self.additinal_args = kwargs

    def __getitem__(self, index):
        image_name = list(self.image_dict.keys())[index]
        if ('classification_name' in self.image_dict[image_name]):
            anno_name = self.image_dict[image_name]['classification_name']
        else:
            anno_name = self.image_dict[image_name]['annotation_name']
        image_dir = self.root_dir+'/'+image_name
        anno_dir = self.root_dir+'/'+anno_name
        image = cv2.cvtColor(cv2.imread(image_dir), cv2.COLOR_BGR2RGB)
        anno_data = readTxt(anno_dir)
        anno_data['altitude'] = self.image_dict[image_name]['height']
        if ('preset_size' in self.additinal_args):
            # pre crop the image to save the resources, note this does not represent the final size of the image
            image, anno_data = random_crop_preset(
                image, anno_data, self.additinal_args['preset_size'])
            # perform data aug that is customized
            image, anno_data = self.cust_transform(image, anno_data)
            # crop image again to ensure the image size is desired
            image, anno_data = random_crop_preset(
                image, anno_data, self.additinal_args['preset_size'])
        else:
            # perform data aug that is customized
            image, anno_data = self.cust_transform(image, anno_data)
        anno_data['bbox'] = torch.from_numpy(
            np.asarray(anno_data['bbox'])).float()
        if (self.torch_transform):
            # perform data aug that within the torchvision lib, note no bbox operation at this stage.
            image = self.torch_transform(image)
        detection_labels = np.ones(len(anno_data['bbox']))
        detection_labels = torch.from_numpy(
            np.asarray(detection_labels)).long()
        anno_data['detection_labels'] = detection_labels
        return image, anno_data

    def __len__(self):
        return len(self.image_dict)


if __name__ == '__main__':
    print('*'*10+'testing')
    root_dir = '/home/zt253/data/WaterfowlDataset/Processed/Bird_G_512Crop'
    csv_dir = '/home/zt253/data/WaterfowlDataset/Processed/Bird_G_512Crop/image_info.csv'
    cust_transform = cust_sequence(
        [RandomHorizontalFlip, RandomVerticalFlip, random_rotate], 0.5)
    torch_transform = None
    task = 'bbox_split_Robert'
    phase = 'train'
    dataset = WaterFowlDataset(root_dir=root_dir, csv_dir=csv_dir, cust_transform=cust_transform,
                               torch_transform=None, task=task, phase=phase, preset_size=512)
    print('length of the dataset', len(dataset))
    from collections import Counter
    altitude = []
    for image, anno in dataset:
        altitude.append(anno['altitude'])
    print (Counter(altitude),sum(Counter(altitude).values()))