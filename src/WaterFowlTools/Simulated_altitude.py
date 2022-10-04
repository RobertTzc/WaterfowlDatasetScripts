"""This script intends to use a lower altitude images to simulate the higher altitude image by downsampling the original images
"""
import os
from tkinter import image_names
import pandas as pd
import cv2
from WaterFowlTools.utils import get_GSD
from WaterFowlTools.anno_util import readTxt
def generate_simulate_images(df,root_dir,ref_Altitude,target_dir):
    os.makedirs(target_dir,exist_ok= True)
    new_df = []
    for idx in range(len(df)):
        item = df.iloc[idx]
        image = cv2.imread(root_dir+'/'+item['image_name'])
        anno_data = readTxt(root_dir+'/'+item['annotation_name'])
        anno_data['altitude'] = item['height']
        GSD,ref_GSD = get_GSD(anno_data,'Pro2',ref_Altitude)
        ratio = GSD/ref_GSD
        print (ratio)
        w,h = image.shape[:2]
        s_w,s_h = int(w*ratio),int(h*ratio)
        bbox = []
        for box in anno_data['bbox']:
            box = [int(i*ratio) for i in box]
            bbox.append(box)
        image_name = item['image_name']
        txt_name = item['annotation_name']
        with open (target_dir+'/'+txt_name,'w') as f:
            for box in bbox:
                #print(box)
                f.writelines('bird,{},{},{},{}\n'.format(box[0],box[1],box[2],box[3]))
        s_image = cv2.resize(image,(s_h,s_w),interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(target_dir+'/'+item['image_name'],s_image)
        new_df.append([image_name,txt_name,ref_Altitude,item['bbox_split_Robert']])
    new_df = pd.DataFrame(new_df)
    new_df.to_csv(target_dir+'/image_info.csv',header = ['image_name','annotation_name','height','bbox_split_Robert'])
if __name__ =='__main__':
    df = pd.read_csv('/home/zt253/data/WaterfowlDataset/Processed/Bird_GwI_15meter/image_info.csv')
    df = df[df['height'].isin([13,14,15,16,17])]
    root_dir  = '/home/zt253/data/WaterfowlDataset/Processed/Bird_GwI_15meter'
    ref_altitude = 45
    generate_simulate_images(df,root_dir,ref_altitude,root_dir+'_simulate_{}_meter'.format(ref_altitude))