import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns
sns.set()

df=pd.read_csv('./../data/ntcir12.csv',sep=',', index_col='Unnamed: 0')

tags=['person','chair', 'book', 'tvmonitor', 'laptop', 'bottle','cup', 'car','diningtable', 'cell phone',
             'keyboard', 'bowl', 'mouse', 'clock','toilet', 'sink', 'remote', 'suitcase', 'pottedplant','refrigerator',
             'knife', 'handbag', 'vase', 'aeroplane', 'cat','bed', 'sofa', 'backpack', 'tie', 'spoon', 'toothbrush',
             'traffic light', 'bicycle', 'train', 'bird', 'microwave', 'bench','fork', 'oven', 'motorbike', 'donut',
             'wine glass', 'pizza','apple', 'scissors', 'umbrella', 'cake', 'bus', 'truck','banana', 'parking meter',
             'sandwich', 'sports ball', 'broccoli','carrot', 'orange', 'teddy bear', 'dog', 'snowboard','skateboard', 'boat',
             'surfboard', 'frisbee', 'skis', 'hot dog','bear', 'elephant', 'toaster', 'stop sign', 'hair drier', 'kite',
             'sheep', 'zebra', 'tennis racket', 'baseball bat', 'fire hydrant','horse', 'cow', 'giraffe', 'baseball glove','day_of_week']

df.index = pd.to_datetime(df.index)

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report,confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#import other features
from sklearn.metrics import accuracy_score

df = df.dropna(how='any',subset=["activity"],axis=0)

df = df.fillna(0)

len(df)

images =df['image_path'].values

def get_median_RGBchannels(f):
    
    image= plt.imread(f)
    #img_pixels= image.flatten()
    red= image[:,:,0]
    green= image[:,:,1]
    blue= image[:,:,2]
    
    return red, green, blue

len(df['image_path'])

column_bins = ['R_bin1', 'R_bin2', 'R_bin3', 'R_bin4', 'R_bin5', 'R_bin6', 'R_bin7', 'R_bin8','G_bin1', 'G_bin2', 'G_bin3', 'G_bin4', 'G_bin5', 'G_bin6', 'G_bin7', 'G_bin8','B_bin1', 'B_bin2', 'B_bin3', 'B_bin4', 'B_bin5', 'B_bin6', 'B_bin7', 'B_bin8']

for i in range(len(column_bins)):
    df[column_bins[i]] = 0

root_file = './../images'

for i,f in enumerate(df['image_path']):

    image= plt.imread(root_file+f)
    red, green, blue = get_median_RGBchannels(root_file+f)

    bin_counts_red, bin_edges_red = np.histogram(red,bins=8) #Red bins
    bin_counts_green, bin_edges_green = np.histogram(green,bins=8) #Green bins
    bin_counts_blue, bin_edges_blue = np.histogram(blue,bins=8) #Blue bins
        
    for j in range(8):
        df[column_bins[j]][i] = bin_counts_red[j]
        df[column_bins[j+8]][i] = bin_counts_green[j]
        df[column_bins[j+16]][i] = bin_counts_blue[j]
    
    print f

plt.hist(red.flatten(),bins=8)
plt.show()

df

df['G_bin7'].max()

maximums = np.zeros(8*3)

for i in range(8*3):
    maximums[i]=df[str(column_bins[i])].max()

maximums

for i in range(8*3):
    df[str(column_bins[i])] = df[str(column_bins[i])]/maximums[i]

df.to_csv('./../data/dataframe_with_RGB_bins.csv', sep='\t')

