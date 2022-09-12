from collections import Counter
from glob import glob
from anno_util import readTxt
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
'''
This script intends to split train/test in terms of IMAGE
Each Image contains multiple class of objects 
and we intended to search a sub-optimal split that can balance all the class as good as we can 
'''


# X, y = make_classification(n_samples=100, weights=[0.94], flip_y=0, random_state=1)
# print(Counter(y))
# # split into train test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=1)
# print(Counter(y_train))
# print(Counter(y_test))



def get_image_dict(root_dir,csv_dir,anno_title = 'classification_name'):
    df = pd.read_csv(csv_dir)
    image_dict = dict()
    for i in range(len(df)):
        image_name = df.iloc[i]['image_name']
        txt_name = df.iloc[i][anno_title]
        altitude = df.iloc[i]['height']
        if (pd.isna(txt_name)):
            pass
        else:
            image_dict[image_name] = dict()
            image_dict[image_name]['category'] = readTxt(root_dir+'/'+txt_name)['category']
            image_dict[image_name]['altitude'] = [altitude]*len(readTxt(root_dir+'/'+txt_name)['category'])
            image_dict[image_name]['bbox'] = readTxt(root_dir+'/'+txt_name)['bbox']
    return image_dict
def get_bins(image_dict,title):
    attribute_list = []
    for image_name in image_dict.keys():
        attribute_list.append(len(image_dict[image_name][title]))
    upper_bound = max(attribute_list)
    lower_bound = min(attribute_list)
    bins = range(lower_bound,upper_bound,(upper_bound-lower_bound)//5)
    return np.asarray(list(bins))

def get_ideal_train_split(image_dict,ratio = 0.8,title = 'category'):
    attribute_list = []
    if (title=='bbox'):# for the bbox we are consider even distributed num of objects into train/test split which is keep density evenly distributed.
        for image_name in image_dict.keys():
            attribute_list.append(len(image_dict[image_name][title]))
        bins = get_bins(image_dict,title)
        cat = np.digitize(attribute_list, bins)
        #print (cat,bins,attribute_list,min(attribute_list),max(attribute_list))
        attribute_list = [bins[i-1] for i in cat]
    else:
        for image_name in image_dict.keys():
            attribute_list.extend(image_dict[image_name][title])
    ct = Counter(attribute_list)
    for key in ct.keys():
        ct[key] = int(ct[key]*0.8+0.2)
    return ct

def calculate_loss(train_list,image_dict,ideal_train_split,title):
    freq = []
    
    if (title=='bbox'):
        bins = get_bins(image_dict,title)
        for image_name in train_list:
            freq.append(len(image_dict[image_name][title]))
        cat = np.digitize(freq, bins)
        freq = [bins[i-1] for i in cat]
    else:
        for image_name in train_list:
            freq.extend(image_dict[image_name][title])
    freq = Counter(freq)
    loss =0
    #print (freq,ideal_train_split)
    
    for key in ideal_train_split.keys():
        loss+=abs(freq.get(key,0)-ideal_train_split[key])/(ideal_train_split[key]+1)
    return loss
loss = float('inf')
combination = []
'''
loss = 1.1830680386594257
combination  =['DJI_0008.jpg', 'DJI_0012.jpg', 'DJI_0013.jpg', 'DJI_0014.jpg', 'DJI_0015.jpg', 'DJI_0018.jpg', 'DJI_0019.jpg', 'DJI_0020.jpg', 'DJI_0022.jpg', 'DJI_0024.jpg', 'DJI_0025.jpg', 'DJI_0026.jpg', 'DJI_0027.jpg', 'DJI_0029.jpg', 'DJI_0030.jpg', 'DJI_0032.jpg', 'DJI_0034.jpg', 'DJI_0035.jpg', 'DJI_0041.jpg', 'DJI_0042.jpg', 'DJI_0043.jpg', 'DJI_0044.jpg', 'DJI_0050.jpg', 'DJI_0051.jpg', 'DJI_0053.jpg', 'DJI_0054.jpg', 'DJI_0055.jpg', 'DJI_0057.jpg', 'DJI_0060.jpg', 'DJI_0086.jpg', 'DJI_0087.jpg', 'DJI_0100.jpg', 'DJI_0101.jpg', 'DJI_0107.jpg', 'DJI_0135.jpg', 'DJI_0158.jpg', 'DJI_0184.jpg', 'DJI_0187.jpg', 'DJI_0214.jpg', 'DJI_0223.jpg', 'DJI_0227.jpg', 'DJI_0239.jpg', 'DJI_0245.jpg', 'DJI_0256.jpg', 'DJI_0258.jpg', 'DJI_0261.jpg', 'DJI_0297.jpg', 'DJI_0300.jpg', 'DJI_0309.jpg', 'DJI_0324.jpg', 'DJI_0325.jpg', 'DJI_0413.jpg', 'DJI_0424.jpg', 'DJI_0429.jpg', 'DJI_0430.jpg', 'DJI_0431.jpg', 'DJI_0433.jpg', 'DJI_0456.jpg', 'DJI_0458.jpg', 'DJI_0465.jpg', 'DJI_0472.jpg', 'DJI_0489.jpg', 'DJI_0498.jpg', 'DJI_0506.jpg', 'DJI_0528.jpg', 'DJI_0562.jpg', 'DJI_0563.jpg', 'DJI_0580.jpg', 'DJI_0595.jpg', 'DJI_0619.jpg', 'DJI_0620.jpg', 'DJI_0631.jpg', 'DJI_0632.jpg', 'DJI_0634.jpg', 'DJI_0648.jpg', 'DJI_0675.jpg', 'DJI_0677.jpg', 'DJI_0678.jpg', 'DJI_0692.jpg', 'DJI_0707.jpg', 'DJI_0710.jpg', 'DJI_0712.jpg', 'DJI_0723.jpg', 'DJI_0724.jpg', 'DJI_0734.jpg', 'DJI_0735.jpg', 'DJI_0740.jpg', 'DJI_0755.jpg', 'DJI_0758.jpg', 'DJI_0784.jpg', 'DJI_0799.jpg', 'DJI_0804.jpg', 'DJI_0822.jpg', 'DJI_0823.jpg', 'DJI_0824.jpg', 'DJI_0839.jpg', 'DJI_0891.jpg', 'DJI_0909.jpg', 'DJI_0999.jpg', 'DJI_1036.jpg', 'DJI_1039.jpg', 'DJI_1042.jpg', 'DJI_1048.jpg', 'DJI_1053.jpg', 'DJI_1128.jpg', 'DJI_1134.jpg', 'DJI_1135.jpg', 'DJI_1648.jpg', 'DJI_1677.jpg', 'DJI_1776.jpg']
'''
@DeprecationWarning
def find_best_split(index,train_list,image_dict,ideal_train_split,title):
    global loss
    global combination
    #build train pools, randomly take images and fill into 
    #print (index,len(image_list),calculate_loss(train_list,image_dict,ideal_split),len(train_list))
    if (index == len(image_list)):
        print ('Reach one result with loss :{}/{}  and num of train images: {}'.format(calculate_loss(train_list,image_dict,ideal_train_split,title),loss,len(train_list)))
        if (calculate_loss(train_list,image_dict,ideal_train_split)<loss):
            loss = calculate_loss(train_list,image_dict,ideal_train_split)
            combination = train_list.copy()
        return 
    else:
        find_best_split(index+1,image_list,train_list.copy(),image_dict)
        train_list.append(image_list[index])
        find_best_split(index+1,image_list,train_list.copy(),image_dict)
        train_list = train_list[:-1]

def random_select(image_list,image_dict,ideal_train_split,epoch,title):
    global loss
    global combination
    wrapped_epoch = tqdm(range(epoch))
    for e in wrapped_epoch:
        train_list = []
        for image_name in image_list:
            if (random.random()<0.8+0.2*(random.random()-0.5)):
                train_list.append(image_name)
        cur_loss = calculate_loss(train_list,image_dict,ideal_train_split,title)
        if (cur_loss<loss):
            loss = cur_loss
            combination = train_list.copy()
        wrapped_epoch.set_description('Epochs:{}, current loss: {}, global minimum: {}'.format(e+1,cur_loss,loss))
    #print (combination)

def visual_distribution(train_list,image_dict,target_dir,title = 'category'):
    train_attribute = []
    test_attribute = []

    if (title == 'bbox'):
        bins = get_bins(image_dict,title)
        for image_name in image_dict.keys():
            if (image_name in train_list):
                train_attribute.append(len(image_dict[image_name][title]))
            else:
                test_attribute.append(len(image_dict[image_name][title]))
        cat  = np.digitize(train_attribute,bins)
        train_attribute = [bins[i-1] for i in cat]
        cat  = np.digitize(test_attribute,bins)
        test_attribute = [bins[i-1] for i in cat]
    else:
        for image_name in image_dict.keys():
            if (image_name in train_list):
                train_attribute.extend(image_dict[image_name][title])
            else:
                test_attribute.extend(image_dict[image_name][title])
    train_ct = Counter(list(train_attribute))
    test_ct = Counter(list(test_attribute))
    attribute_names = Counter(train_attribute+test_attribute)
    for name in attribute_names.keys():
        train_ct[name] = train_ct.get(name,0)
        test_ct[name] = test_ct.get(name,0)
    plt.figure()
    plt.title('split dataset visualization based on {}'.format(title))
    plt.bar([str(i) for i in sorted(attribute_names.keys())],[train_ct.get(name,0) for name in sorted(attribute_names.keys())],label = 'train_set')
    plt.bar([str(i) for i in sorted(attribute_names.keys())],[test_ct.get(name,0) for name in sorted(attribute_names.keys())],label = 'test_set')
    plt.legend()
    plt.savefig(target_dir)
    plt.show()
    
if __name__ == '__main__':
    #title = 'altitude'
    title = 'bbox'
    folder_name = 'Bird_I'
    #anno_title = 'classification_name'
    anno_title = 'annotation_name'
    image_dict= get_image_dict(root_dir = '/home/zt253/data/WaterfowlDataset/Processed/{}'.format(folder_name),
                               csv_dir ='/home/zt253/data/WaterfowlDataset/Processed/{}/image_info.csv'.format(folder_name),
                               anno_title = anno_title)
    image_list = list(image_dict.keys())
    ideal_train_split = get_ideal_train_split(image_dict = image_dict,ratio = 0.8,title = title)
    print ('ideal_split',ideal_train_split)
    print ('*'*10,'staring finding the best split',len(image_list))
    #find_best_split(0,image_list,[],image_dict)
    random_select(image_list = image_list,image_dict = image_dict,ideal_train_split = ideal_train_split,epoch = 10000,title = title)
    print ('*'*20,'Final result of train split:',len(combination))
    print (combination)

    
    visual_distribution(train_list = combination,
                        image_dict = image_dict,
                        target_dir='/home/zt253/data/WaterfowlDataset/WaterfowlDatasetScripts/Split_data/{}_split_visualization_basedOn_{}.jpg'.format(folder_name,title),
                        title = title)
    
    import json
    with open('/home/zt253/data/WaterfowlDataset/WaterfowlDatasetScripts/Split_data/{}_split_basedOn_{}.json'.format(folder_name,title),'w') as f:
        json.dump(combination,f)