import pandas as pd

'''
Here we will apply Robert's split on each dataset
We based on class of the targets(if exists) to split the train and test
and this split is NOT guarantee to be the best split
we did 100000 iterations to ensure its a good split
'''
def write_split(csv_dir,train_split):
    df = pd.read_csv(csv_dir)
    robert_split = []
    for i in range(len(df)):
        image_name = df.iloc[i]['image_name']
        if (image_name in train_split):
            robert_split.append('train')
        else:
            robert_split.append('test')
    df['robert_split'] = robert_split
    df.to_csv(csv_dir,index = False)

Dataset_I_Split = ['DJI_0008.jpg', 'DJI_0012.jpg', 'DJI_0013.jpg', 'DJI_0014.jpg', 'DJI_0015.jpg', 'DJI_0018.jpg', 'DJI_0019.jpg', 'DJI_0020.jpg', 'DJI_0022.jpg', 'DJI_0024.jpg', 'DJI_0025.jpg', 'DJI_0026.jpg', 'DJI_0027.jpg', 'DJI_0029.jpg', 'DJI_0030.jpg', 'DJI_0032.jpg', 'DJI_0034.jpg', 'DJI_0035.jpg', 'DJI_0041.jpg', 'DJI_0042.jpg', 'DJI_0043.jpg', 'DJI_0044.jpg', 'DJI_0050.jpg', 'DJI_0051.jpg', 'DJI_0053.jpg', 'DJI_0054.jpg', 'DJI_0055.jpg', 'DJI_0057.jpg', 'DJI_0060.jpg', 'DJI_0086.jpg', 'DJI_0087.jpg', 'DJI_0100.jpg', 'DJI_0101.jpg', 'DJI_0107.jpg', 'DJI_0135.jpg', 'DJI_0158.jpg', 'DJI_0184.jpg', 'DJI_0187.jpg', 'DJI_0214.jpg', 'DJI_0223.jpg', 'DJI_0227.jpg', 'DJI_0239.jpg', 'DJI_0245.jpg', 'DJI_0256.jpg', 'DJI_0258.jpg', 'DJI_0261.jpg', 'DJI_0297.jpg', 'DJI_0300.jpg', 'DJI_0309.jpg', 'DJI_0324.jpg', 'DJI_0325.jpg', 'DJI_0413.jpg', 'DJI_0424.jpg', 'DJI_0429.jpg', 'DJI_0430.jpg', 'DJI_0431.jpg', 'DJI_0433.jpg', 'DJI_0456.jpg', 'DJI_0458.jpg', 'DJI_0465.jpg', 'DJI_0472.jpg', 'DJI_0489.jpg', 'DJI_0498.jpg', 'DJI_0506.jpg', 'DJI_0528.jpg', 'DJI_0562.jpg', 'DJI_0563.jpg', 'DJI_0580.jpg', 'DJI_0595.jpg', 'DJI_0619.jpg', 'DJI_0620.jpg', 'DJI_0631.jpg', 'DJI_0632.jpg', 'DJI_0634.jpg', 'DJI_0648.jpg', 'DJI_0675.jpg', 'DJI_0677.jpg', 'DJI_0678.jpg', 'DJI_0692.jpg', 'DJI_0707.jpg', 'DJI_0710.jpg', 'DJI_0712.jpg', 'DJI_0723.jpg', 'DJI_0724.jpg', 'DJI_0734.jpg', 'DJI_0735.jpg', 'DJI_0740.jpg', 'DJI_0755.jpg', 'DJI_0758.jpg', 'DJI_0784.jpg', 'DJI_0799.jpg', 'DJI_0804.jpg', 'DJI_0822.jpg', 'DJI_0823.jpg', 'DJI_0824.jpg', 'DJI_0839.jpg', 'DJI_0891.jpg', 'DJI_0909.jpg', 'DJI_0999.jpg', 'DJI_1036.jpg', 'DJI_1039.jpg', 'DJI_1042.jpg', 'DJI_1048.jpg', 'DJI_1053.jpg', 'DJI_1128.jpg', 'DJI_1134.jpg', 'DJI_1135.jpg', 'DJI_1648.jpg', 'DJI_1677.jpg', 'DJI_1776.jpg']

if __name__ == '__main__':
    csv_dir = '/home/zt253/data/WaterfowlDataset/Processed/Bird_I/image_info.csv'
    write_split(csv_dir,Dataset_I_Split)