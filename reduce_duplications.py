#This script intends to be eliminate duplicate and redudant boudning box
#after merging the labeled cropped image to the original image
#It takes the txt file as input and regenerate the txt file after filtering theb unwanted bounding box
import glob
import numpy as np


def py_cpu_nms(dets,thresh):  
    """Pure Python NMS baseline.""" 
    dets = np.asarray(dets) 
    x1 = np.asarray([float(i) for i in dets[:, 0]]) 
    y1 = np.asarray([float(i) for i in dets[:, 1]]) 
    x2 = np.asarray([float(i) for i in dets[:, 2]])   
    y2 = np.asarray([float(i) for i in dets[:, 3]])  
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  
    order = areas.argsort()[::-1]  
    keep = []  
    while order.size > 0:  
        i = order[0]  
        keep.append(i)  
        xx1 = np.maximum(x1[i], x1[order[1:]])  
        yy1 = np.maximum(y1[i], y1[order[1:]])  
        xx2 = np.minimum(x2[i], x2[order[1:]])  
        yy2 = np.minimum(y2[i], y2[order[1:]])  
        w = np.maximum(0.0, xx2 - xx1 + 1)  
        h = np.maximum(0.0, yy2 - yy1 + 1)  
        inter = w * h  
        ovr = 1.*inter / (areas[i] + areas[order[1:]] - inter)  
        inds = np.where((ovr <= thresh) & (areas[order[1:]]>1.2*inter))[0]  
        order = order[inds + 1]  



  
    return keep

if __name__ == '__main__':

    from anno_util import readTxt
    txt_list =  glob.glob('/home/zt253/data/WaterfowlDataset/Processed/Bird_*/*.txt')
    txt_list = [i for i in txt_list if ('mask' not in i)]
    for txt_dir in txt_list:
        print (txt_dir)
        re = readTxt(txt_dir)
        if (re['bbox']):
            
            keep = (py_cpu_nms(re['bbox'],0.25))
            with open(txt_dir,'w') as f:
                for box,cat in zip(np.asarray(re['bbox'])[keep],np.asarray(re['category'])[keep]):
                    f.writelines('{},{},{},{},{}\n'.format(cat,box[0],box[1],box[2],box[3]))
    
    '''
    #Unit Test
    from anno_util import *
    txt_dir ='/home/zt253/data/WaterfowlDataset/Processed/Bird_J/Cloud_MoistSoil_30m_DJI_0522.txt'
    target_dir = txt_dir.replace('Bird_J','Bird_X')
    re = readTxt(txt_dir)
    if (re['bbox']):
        
        keep = (py_cpu_nms(re['bbox'],0.25))
        with open(target_dir,'w') as f:
            for box,cat in zip(np.asarray(re['bbox'])[keep],np.asarray(re['category'])[keep]):
                f.writelines('{},{},{},{},{}\n'.format(cat,box[0],box[1],box[2],box[3]))
    '''