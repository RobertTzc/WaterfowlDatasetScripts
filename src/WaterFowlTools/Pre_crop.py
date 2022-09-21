from anno_util import writeTxt
from PIL import Image
from logging import root
from re import L
from unicodedata import category
import cv2
import pandas as pd
import numpy as np
import os
import math
from WaterFowlTools.anno_util import readTxt


def crop_image_split_pos_neg(image, bbox, overlap, size):
    w, h = image.size
    coor_pos_list = []
    coor_neg_list = []
    for i in range(math.ceil(w*(1-overlap)/size)):
        for j in range(math.ceil(h*(1-overlap)/size)):
            coord = ([int(size*(1-overlap)*i), int(size*(1-overlap)*j)])
            w_range = [size*(1-overlap)*i, size*(1-overlap)*i+size]
            h_range = [size*(1-overlap)*j, size*(1-overlap)*j+size]
            for box in bbox:
                y1, x1, y2, x2 = box
                if (y1 > w_range[0] and y1 < w_range[1] and x1 > h_range[0] and x1 < h_range[1] or  # left up corner in range
                        y2 > w_range[0] and y2 < w_range[1] and x2 > h_range[0] and x2 < h_range[1]):  # right bottom corner in range
                    coor_pos_list.append(coord)
                    break
            if (coord not in coor_pos_list):
                coor_neg_list.append(coord)
    return coor_pos_list, coor_neg_list


class WaterFowlDataset_crop(object):
    def __init__(self, root_dir, csv_dir):
        self.df = pd.read_csv(csv_dir)
        self.root_dir = root_dir
        self.image_dict = dict()
        for idx in range(len(self.df)):
            item = self.df.iloc[idx]
            self.image_dict[item['image_name']] = dict()
            for keys in item.keys():
                if (keys == 'image_name'):
                    pass
                else:
                    self.image_dict[item['image_name']][keys] = item[keys]

    def side_by_side_crop(self, crop_size=2048, overlap=0.2, target_dir=None,has_classification = False):
        assert target_dir, 'target_dir cannot be None'
        os.makedirs(target_dir, exist_ok=True)
        new_df = []
        for image_name in self.image_dict.keys():
            keys = sorted(self.image_dict[image_name].keys())
            keys.remove('annotation_name')
            keys.remove('index')
            if (has_classification):
                keys.remove('classification_name')
                anno_dir = self.root_dir+'/' + \
                self.image_dict[image_name]['classification_name']
            else:
                anno_dir = self.root_dir+'/' + \
                self.image_dict[image_name]['annotation_name']
            print('*'*10, 'keeping the following attributes', keys)
            image = Image.open(self.root_dir+'/{}'.format(image_name))
            #print (image,self.root_dir+'/{}'.format(image_name))
            
            bbox = readTxt(anno_dir)['bbox']
            category = readTxt(anno_dir)['category']
            coor_pos_list, coor_neg_list = crop_image_split_pos_neg(
                image, bbox, overlap=overlap, size=crop_size)
            for coord in coor_pos_list:
                coord = [int(coord[0]), int(coord[1])]
                sub_image = image.crop(
                    (coord[0], coord[1], coord[0]+crop_size, coord[1]+crop_size)).resize((crop_size, crop_size))
                #sub_image = image[coord[0]:coord[0]+crop_size,coord[1]:coord[1]+crop_size,:]
                sub_bbox = []
                sub_category = []
                w_range = [coord[0], coord[0]+crop_size]
                h_range = [coord[1], coord[1]+crop_size]
                for cat, box in zip(category, bbox):
                    y1, x1, y2, x2 = box
                    # left up corner in range
                    if (y1 > w_range[0] and y1 < w_range[1] and x1 > h_range[0] and x1 < h_range[1]):
                        s_box = [box[0]-coord[0], box[1]-coord[1],
                                 box[2]-coord[0], box[3]-coord[1]]
                        sub_bbox.append(s_box)
                        sub_category.append(cat)
                    # right bottom corner in range
                    elif (y2 > w_range[0] and y2 < w_range[1] and x2 > h_range[0] and x2 < h_range[1]):
                        s_box = [box[0]-coord[0], box[1]-coord[1],
                                 box[2]-coord[0], box[3]-coord[1]]
                        sub_bbox.append(s_box)
                        sub_category.append(cat)
                sub_image_name = '{}_{}_{}.{}'.format(image_name.split(
                    '.')[0], coord[0], coord[1], image_name.split('.')[1])
                sub_image.save(target_dir+'/{}'.format(sub_image_name))
                if(has_classification):
                    writeTxt(
                        target_dir+'/{}_class.txt'.format(sub_image_name.split('.')[0]), sub_category, sub_bbox)
                    writeTxt(
                        target_dir+'/{}.txt'.format(sub_image_name.split('.')[0]), ['bird' for _ in sub_category], sub_bbox)
                    new_item = [sub_image_name,sub_image_name.split('.')[0]+'.txt',sub_image_name.split('.')[0]+'_class.txt']
                else:
                    writeTxt(
                        target_dir+'/{}.txt'.format(sub_image_name.split('.')[0]), sub_category, sub_bbox)
                    new_item = [sub_image_name,sub_image_name.split('.')[0]+'.txt']
                for key in keys:
                    new_item += [self.image_dict[image_name][key]]
                new_df.append(new_item)

        if (has_classification): 
            keys = ['image_name','annotation_name','classification_name']+keys
        else: 
            keys = ['image_name','annotation_name']+keys
        new_df = pd.DataFrame(new_df)
        new_df.to_csv(target_dir+'/image_info.csv', header=keys)


if __name__ == '__main__':
    dataset = WaterFowlDataset_crop(root_dir='/home/zt253/data/WaterfowlDataset/Processed/Bird_D',
                                    csv_dir='/home/zt253/data/WaterfowlDataset/Processed/Bird_D/image_info.csv'
                                    )
    dataset.side_by_side_crop(crop_size=512, overlap=0.2,
                              target_dir='/home/zt253/data/WaterfowlDataset/Processed/Bird_D_512Crop')
