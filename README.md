# WaterfowlDatasetScripts
The repository contains all necessary scripts for Waterfowl dataset maintainence and utlization
  anno_util.py:
    necessary util funcs for read and operate dataset
  Reduce_duplications.py
    Designed to solve the redudant bounding box caused by cropped image labeling issue
  split_dataset_classification.py
    using loss func to find an optimal split point to split the dataset with balanced split on train and test distribution on each class
  WriteTrainTestSplit.py
    used to write an calculated split to applied on an existing image list.
