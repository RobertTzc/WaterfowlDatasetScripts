import numpy as np
import pandas as pd
from collections import Counter
def GSD_calculation(image,anno_data,drone_type = 'Pro2'):
    height = anno_data['altitude']
    if (drone_type == 'Pro2'):
        ref_GSD = (13.2 * 90)/(10.26*5472)
        GSD=(13.2 * height)/(10.26*5472)
    elif (drone_type == 'Air2'):
        ref_GSD = (6.4*90)/(4.3*8000)
        GSD = (6.4*height)/(4.3*8000)
    else:
        ref_GSD = (13.2 * 90)/(10.26*5472)
        GSD = (13.2 * 90)/(10.26*5472)
    return GSD,ref_GSD

def readTxt(txt_dir):
    with open(txt_dir,'r') as f:
        data = f.readlines()
    bbox = []
    category = []
    for line in data:
        line = line.replace('\n','').split(',')
        bbox.append([int(i) for i in line[1:]])
        category.append(line[0])
    re = {'bbox':bbox,'category':category}
    return re
def find_dominant_class(txt_dir,title = 'category'):
    re = readTxt(txt_dir)
    ct = Counter(re[title])
    dominant_ct = ct.most_common(1)[0]
    return dominant_ct,ct

def assign_image_based_label(root_dir,csv_dir):
    df = pd.read_csv(csv_dir)
    image_list = []
    anno_list = []
    for i in range(len(df)):
        image_name = df.iloc[i]['image_name']
        txt_name = df.iloc[i]['classification_name']
        try:
            dominant_ct,ct = find_dominant_class(root_dir+'/'+txt_name)
            image_list.append(image_name)
            anno_list.append(dominant_ct[0])
        except:
            pass
    return image_list,anno_list