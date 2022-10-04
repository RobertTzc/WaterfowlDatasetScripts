import cv2
import pandas as pd
import os
import glob
#returns image 
def waterfowlDataset(root,info_csv,isVisual=False,target_dir = ''):
    if (isVisual == True):
        assert(target_dir)
    df = pd.read_csv(info_csv)
    os.makedirs(target_dir,exist_ok=True)
    for index in range(len(df)):
        image_dir = root+'/{}'.format(df.iloc[index]['image_name'])
        if('classification_name' in df.columns and not pd.isna(df.iloc[index]['classification_name'])):
            anno_dir = root+'/{}'.format(df.iloc[index]['classification_name'])
        else:
            anno_dir = root+'/{}'.format(df.iloc[index]['annotation_name'])
        image = cv2.imread(image_dir)
        with open(anno_dir,'r') as f:
            anno_data = f.readlines()
        bbox = []
        for line in anno_data:
            line = line.replace('\n','').split(',')
            bbox.append([line[0],int(line[1]),int(line[2]),int(line[3]),int(line[4])])
        for box in bbox:
            if (box[2]<0 or box[1]<0 or box[4]>image.shape[0] or box[3]>image.shape[1]):
                print ('warning notice bbox',box)
            if ('discard' in box[0]):
                color = (255,255,255)
                print (image_dir.split('/')[-1])
                width = 1
            else:
                color = (255,0,0)
                
                width = 1
            image = cv2.rectangle(image,(box[1],box[2]),(box[3],box[4]),color,width)
            image = cv2.putText(image,box[0],(box[1],box[2]),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,width)
        cv2.imwrite(target_dir+'/'+image_dir.split('/')[-1],image)
        
if __name__ == '__main__':
    folder_list  = sorted(glob.glob('/home/zt253/data/WaterfowlDataset/Processed/Bird_G_simulate_90_meter*'))
    for folder_dir in folder_list:
        print (folder_dir)
        waterfowlDataset(folder_dir,folder_dir+'/image_info.csv',True,folder_dir.replace('Processed','Visual'))