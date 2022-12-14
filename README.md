# WaterfowlDatasetScripts

This repository contains all necessary scripts for Waterfowl dataset maintainence and utlization
## To use the default dataset builder:
      from WaterFowlTools.Dataset import WaterFowlDataset
      from WaterFowlTools import cust_transform
      class WaterFowlCustomDataset(WaterFowlDataset):
      ... <(optioinal) re-implement __init__(),__getitem()__()>
#### args:
        root_dir: specify the location image stores
        csv_dir: specify where the corresponding image_info. 
        csv file stores, usually within the same root dir
        cust_transform: cust transformations that can be used for image & bbox aug see details in next paragraph
        torch_transform:default transformations from torchvision
        task: which split this dataset is using current support :'altitude_split_Robert','bbox_split_Robert' etc
        phase :'train' or 'test'
        **kwargs: extra args that are taken in input as a dict format.
#### EXAMPLE
        from WaterFowlTools.Dataset import WaterFowlDataset
        from WaterFowlTools import cust_transform
        class WaterFowlDatasetRetinanet(WaterFowlDataset):
          def __init__(self,root_dir,csv_dir,cust_transform,torch_transform,task = 'altitude_split_Robert',phase = 'train',**kwargs):
              super().__init__(root_dir = root_dir,csv_dir = csv_dir,cust_transform=cust_transform,torch_transform = torch_transform,task = task,phase = phase,**kwargs)
              self.encoder = DataEncoder()
              
          def __getitem__(self,index):
              image,anno_data = super().__getitem__(index)
User can reuse the base WaterFowlDataset or simply use the Default dataset to load data with format of {image(torch Tensor),anno_data(dict)} anno_data contains keys ['bbox', 'detection_labels', 'altitude'] for each correspoing image
> It is recommend that you inherit this class without directly use of the default class even you dont want change any thing.
## cust_transform.py 
### contains some custom transformantions that is not available in torchvision(mostly image+bbox transformation) also comes with altitude based scaling
    from WaterFowlTools import cust_transform
    #this will form a sequence to all transformations
    cust_transform = cust_transform.cust_sequence([
        cust_transform.RandomHorizontalFlip,
        cust_transform.RandomVerticalFlip,
        cust_transform.random_rotate, #random rotate images [0,180]
        altitude_based_scale,]#altitude based scale the image, default ref 60 meters
        p = 0.5) #probability of each transformation, could be list, must match the size of transforms
#### Other funcs:
      altitude_based_scale(image, anno_data, ref_altitude=60)#scale the image ref to a preset altitude


## anno_util.py:
Contains necessary util funcs for read and operate dataset
```
readTxt(txt_dir)
  Given the txt directory, return the annotation in dict format contains keys: ['bbox', 'category']
```

## Reduce_duplications.py
    Designed to solve the redudant bounding box caused by cropped image labeling issue
## split_dataset_classification.py
    using loss func to find an optimal split point to split the dataset with balanced split on train and test distribution on each class
## WriteTrainTestSplit.py
    used to write an calculated split to applied on an existing image list.


## util.py
```
py_cpu_nms(dets,thresh=0.25)
  use NMS to filter heavily overlapped images, default IOU threshold is set to be 0.25, Also adding extra step to remove tiny bbox that lay inside the large bbox that cannot be filtered by the IOU thresh
```
```
get_image_taking_conditions(image_dir)
  given an image_dir, read the image meta data(if exists) and return a dict contains ISO,shutter,aperture,image_name,altitude info.
```
```
get_sub_image(mega_image,overlap=0.2,ratio=1,cropSize = 512)
  crop the original image based on overlap, ratio and crop size
the actual size of the cropped images will be ratio * cropSize to be adaptive to the alti
```
```
get_GSD(anno_data, camera_type='Pro2', ref_altitude=60)
  Calculate the GSD(Ground sampling distance of the given image and alitude)
return:
  The actual GSD of the image and at 60m the reference GSD.
```
## mAp.py
```
mAp_calculate(image_name_list, gt_txt_list, pred_txt_list, iou_thresh=0.3)
  calculate the mAP of given list of predictions, note the order of image_name_list,gt_txt_list,pred_txt_list should all be the same inorder to have reliable result
  return: precision, recall, sum_AP, mrec, mprec, area
```

## Requirements

```
python>=3.8
cv2
pandas
numpy
matplotlib
PILLOW
sklearn
tqdm
pytorch>=11.1
```