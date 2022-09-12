'''
This scripts will applied the split into the image_info.csv files for later usage.
'''
from operator import index
import pandas as pd
import json

root_dir  = '/home/zt253/data/WaterfowlDataset/Processed'
split_dir = '/home/zt253/data/WaterfowlDataset/WaterfowlDatasetScripts/Split_data'
split_dict = {
    'Bird_A':['altitude','bbox'],
    'Bird_A':['altitude','bbox'],
    'Bird_B':['altitude','bbox'],
    'Bird_C':['altitude','bbox'],
    'Bird_D':['bbox'],
    'Bird_E':['bbox'],
    'Bird_F':['bbox'],
    'Bird_G':['altitude','bbox'],
    'Bird_H':['altitude','bbox','category'],
    'Bird_I':['bbox','category'],
    'Bird_J':['altitude','bbox'],
}
for folder_name in split_dict.keys():
    folder_dir = root_dir+'/'+folder_name
    df = pd.read_csv(folder_dir+'/image_info.csv',index_col=False)
    for task in split_dict[folder_name]:
        with open(split_dir+'/{}_split_basedOn_{}.json'.format(folder_name,task),'r') as f:
            train_split = json.load(f)
        new_col = []
        for i in range(len(df)):
            if (df.iloc[i]['image_name'] in train_split):
                new_col.append('train')
            else:
                new_col.append('test')
        df['{}_split_Robert'.format(task)] = new_col
    df=df.rename(columns = {'Unnamed: 0':'index'})
    df.to_csv(folder_dir+'/image_info.csv',index=False)
