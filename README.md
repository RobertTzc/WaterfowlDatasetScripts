# WaterfowlDatasetScripts
## The repository contains all necessary scripts for Waterfowl dataset maintainence and utlization
### To use these packages:
      from WaterFowlTools.Dataset import WaterFowlDataset
      from WaterFowlTools import cust_transform
      class WaterFowlCustomDataset(WaterFowlDataset):
      ... <(optioinal) re-implement __init__(),__getitem()__()>
#### User can reuse the base WaterFowlDataset or simply use the Default dataset to load data with format of {image(torch Tensor),anno_data(dict)} anno_data contains keys ['bbox','detection_labels','altitude'] for each correspoing image
#### It is recommend that you inherit this class without directly use of the default class even you dont want change any thing.
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
<!-- ### anno_util.py:
necessary util funcs for read and operate dataset
  #### func: GSD_calculation(image,anno_data,drone_type = 'Pro2')
  Reduce_duplications.py
    Designed to solve the redudant bounding box caused by cropped image labeling issue
  split_dataset_classification.py
    using loss func to find an optimal split point to split the dataset with balanced split on train and test distribution on each class
  WriteTrainTestSplit.py
    used to write an calculated split to applied on an existing image list. -->
