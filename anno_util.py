import numpy as np
import pandas as pd
from collections import Counter

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