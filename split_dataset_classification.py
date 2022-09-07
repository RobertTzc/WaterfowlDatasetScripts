from collections import Counter
from glob import glob
from anno_util import readTxt
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

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



def get_image_dict_classification(root_dir,csv_dir):
    df = pd.read_csv(csv_dir)
    image_dict = dict()
    for i in range(len(df)):
        image_name = df.iloc[i]['image_name']
        txt_name = df.iloc[i]['classification_name']
        if (pd.isna(txt_name)):
            pass
        else:
            image_dict[image_name] = readTxt(root_dir+'/'+txt_name)['category']
    return image_dict

def get_ideal_split_classification(image_dict,ratio = 0.8):
    categories = []
    for v in image_dict.values():
        categories.extend(v)
    ct = Counter(categories)
    for key in ct.keys():
        ct[key] = int(ct[key]*0.8)
    return ct

def calculate_loss(train_list,image_dict,ideal_split):
    freq = []
    for image_name in train_list:
        freq.extend(image_dict[image_name])
    freq = Counter(freq)
    loss =0
    for key in ideal_split.keys():
        loss+=abs(freq.get(key,0)-ideal_split[key])/(ideal_split[key]+1)
    return loss

loss = 1.1830680386594257
combination  =['DJI_0008.jpg', 'DJI_0012.jpg', 'DJI_0013.jpg', 'DJI_0014.jpg', 'DJI_0015.jpg', 'DJI_0018.jpg', 'DJI_0019.jpg', 'DJI_0020.jpg', 'DJI_0022.jpg', 'DJI_0024.jpg', 'DJI_0025.jpg', 'DJI_0026.jpg', 'DJI_0027.jpg', 'DJI_0029.jpg', 'DJI_0030.jpg', 'DJI_0032.jpg', 'DJI_0034.jpg', 'DJI_0035.jpg', 'DJI_0041.jpg', 'DJI_0042.jpg', 'DJI_0043.jpg', 'DJI_0044.jpg', 'DJI_0050.jpg', 'DJI_0051.jpg', 'DJI_0053.jpg', 'DJI_0054.jpg', 'DJI_0055.jpg', 'DJI_0057.jpg', 'DJI_0060.jpg', 'DJI_0086.jpg', 'DJI_0087.jpg', 'DJI_0100.jpg', 'DJI_0101.jpg', 'DJI_0107.jpg', 'DJI_0135.jpg', 'DJI_0158.jpg', 'DJI_0184.jpg', 'DJI_0187.jpg', 'DJI_0214.jpg', 'DJI_0223.jpg', 'DJI_0227.jpg', 'DJI_0239.jpg', 'DJI_0245.jpg', 'DJI_0256.jpg', 'DJI_0258.jpg', 'DJI_0261.jpg', 'DJI_0297.jpg', 'DJI_0300.jpg', 'DJI_0309.jpg', 'DJI_0324.jpg', 'DJI_0325.jpg', 'DJI_0413.jpg', 'DJI_0424.jpg', 'DJI_0429.jpg', 'DJI_0430.jpg', 'DJI_0431.jpg', 'DJI_0433.jpg', 'DJI_0456.jpg', 'DJI_0458.jpg', 'DJI_0465.jpg', 'DJI_0472.jpg', 'DJI_0489.jpg', 'DJI_0498.jpg', 'DJI_0506.jpg', 'DJI_0528.jpg', 'DJI_0562.jpg', 'DJI_0563.jpg', 'DJI_0580.jpg', 'DJI_0595.jpg', 'DJI_0619.jpg', 'DJI_0620.jpg', 'DJI_0631.jpg', 'DJI_0632.jpg', 'DJI_0634.jpg', 'DJI_0648.jpg', 'DJI_0675.jpg', 'DJI_0677.jpg', 'DJI_0678.jpg', 'DJI_0692.jpg', 'DJI_0707.jpg', 'DJI_0710.jpg', 'DJI_0712.jpg', 'DJI_0723.jpg', 'DJI_0724.jpg', 'DJI_0734.jpg', 'DJI_0735.jpg', 'DJI_0740.jpg', 'DJI_0755.jpg', 'DJI_0758.jpg', 'DJI_0784.jpg', 'DJI_0799.jpg', 'DJI_0804.jpg', 'DJI_0822.jpg', 'DJI_0823.jpg', 'DJI_0824.jpg', 'DJI_0839.jpg', 'DJI_0891.jpg', 'DJI_0909.jpg', 'DJI_0999.jpg', 'DJI_1036.jpg', 'DJI_1039.jpg', 'DJI_1042.jpg', 'DJI_1048.jpg', 'DJI_1053.jpg', 'DJI_1128.jpg', 'DJI_1134.jpg', 'DJI_1135.jpg', 'DJI_1648.jpg', 'DJI_1677.jpg', 'DJI_1776.jpg']

def find_best_split(index,image_list,train_list,image_dict,ideal_split):
    global loss
    global combination
    #build train pools, randomly take images and fill into 
    #print (index,len(image_list),calculate_loss(train_list,image_dict,ideal_split),len(train_list))
    if (index == len(image_list)):
        print ('Reach one result with loss :{}/{}  and num of train images: {}'.format(calculate_loss(train_list,image_dict,ideal_split),loss,len(train_list)))
        if (calculate_loss(train_list,image_dict,ideal_split)<loss):
            loss = calculate_loss(train_list,image_dict,ideal_split)
            combination = train_list.copy()
        return 
    else:
        find_best_split(index+1,image_list,train_list.copy(),image_dict)
        train_list.append(image_list[index])
        find_best_split(index+1,image_list,train_list.copy(),image_dict)
        train_list = train_list[:-1]

def random_select(image_list,image_dict,ideal_split,epoch):
    global loss
    global combination
    wrapped_epoch = tqdm(range(epoch))
    for e in wrapped_epoch:
        train_list = []
        for image_name in image_list:
            if (random.random()<0.8):
                train_list.append(image_name)
        cur_loss = calculate_loss(train_list,image_dict,ideal_split)
        if (cur_loss<loss):
            loss = cur_loss
            combination = train_list.copy()
        wrapped_epoch.set_description('Epochs:{}, current loss: {}, global minimum: {}'.format(e+1,cur_loss,loss))
    print (combination)

def visual_distribution(train_list,image_dict,target_dir):
    train_category = []
    test_category = []
    for key in image_dict.keys():
        if (key in train_list):
            train_category.extend(image_dict[key])
        else:
            test_category.extend(image_dict[key])
    train_ct = Counter(train_category)
    test_ct = Counter(test_category)
    category_names = Counter(train_category+test_category)
    for name in category_names.keys():
        train_ct[name] = train_ct.get(name,0)
        test_ct[name] = test_ct.get(name,0)
    plt.figure()
    plt.title('split dataset visualization')
    plt.bar(list(category_names.keys()),[train_ct.get(name,0) for name in category_names.keys()],label = 'train_set')
    plt.bar(list(category_names.keys()),[test_ct.get(name,0) for name in category_names.keys()],label = 'test_set')
    plt.legend()
    plt.savefig(target_dir)
    plt.show()
    
if __name__ == '__main__':
    image_dict= get_image_dict_classification('/home/zt253/data/WaterfowlDataset/Processed/Bird_I',
                                                    '/home/zt253/data/WaterfowlDataset/Processed/Bird_I/image_info.csv')
    image_list = list(image_dict.keys())
    ideal_split = get_ideal_split_classification(image_dict,0.8)
    print ('*'*10,'staring finding the best split',len(image_list))
    #find_best_split(0,image_list,[],image_dict)
    random_select(image_list,image_dict,ideal_split,10)
    print ('*'*20,'Final result of train split:')
    print (combination)
    visual_distribution(combination,image_dict,'/home/zt253/data/WaterfowlDataset/Scripts/Bird_I_split_visualization.jpg')