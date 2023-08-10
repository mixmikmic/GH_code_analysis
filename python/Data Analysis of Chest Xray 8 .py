import torch
from skimage import io
import seaborn as sns

import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from PIL import Image
import cv2
import time
import numpy as np
import torchvision.models as models
from torch.optim import lr_scheduler
import torchvision
import torch.nn as nn
from torchvision import datasets, models, transforms,utils
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.autograd import Variable
import os
import pandas as pd
import glob
from skimage import io
import matplotlib.pyplot as plt
import os 

get_ipython().magic('matplotlib inline')
use_gpu = torch.cuda.is_available()

# Config
DATA_DIR = 'data'
TRAIN_DIR = 'train_3_channel'
TEST_DIR = 'test'
VAL_DIR = 'val_3_channel'
TRAIN_DS_LOC = os.path.join(DATA_DIR, TRAIN_DIR)
TEST_DS_LOC = os.path.join(DATA_DIR, TEST_DIR)
VAL_DS_LOC = os.path.join(DATA_DIR, VAL_DIR)
IMG_PATTERN = '*.png'
IMG_PROPERTIES_FILE = 'Data_Entry_2017.csv'
LABEL_SEP = '|'
Height=512
Width=512


# Retrieve and process labels
l2i = {'Atelectasis': 0,
 'Cardiomegaly': 1,
 'Consolidation': 2,
 'Edema': 3,
 'Effusion': 4,
 'Emphysema': 5,
 'Fibrosis': 6,
 'Hernia': 7,
 'Infiltration': 8,
 'Mass': 9,
 'No Finding': 10,
 'Nodule': 11,
 'Pleural_Thickening': 12,
 'Pneumonia': 13,
 'Pneumothorax': 14}

properties = pd.read_csv(IMG_PROPERTIES_FILE, skiprows=1, header=None, low_memory=False, na_filter=False).values
labels = {prop[0] : [ l2i[label] for label in  prop[1].split(LABEL_SEP)] for prop in properties}

num_classes=len(l2i)

# Data transforms
data_transforms = {
    'train': transforms.Compose([
         # transforms.ToPILImage(),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    
    ]),
    'val': transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),

    ]),
}

#import image data set description
df = pd.read_csv('Data_Entry_2017.csv')
df.head()

df = df[['Image Index','Finding Labels','Follow-up #','Patient ID','Patient Age','Patient Gender']]

#create new columns for each decease
pathology_list = ['Cardiomegaly','Emphysema','Effusion','Hernia','Nodule','Pneumothorax','Atelectasis','Pleural_Thickening','Mass','Edema','Consolidation','Infiltration','Fibrosis','Pneumonia']

for pathology in pathology_list :
    df[pathology] = df['Finding Labels'].apply(lambda x: 1 if pathology in x else 0)

#remove Y after age
df['Age']=df['Patient Age'].apply(lambda x: x[:-1]).astype(int)

plt.figure(figsize=(15,10))
gs = gridspec.GridSpec(8,1)
ax1 = plt.subplot(gs[:7, :])
ax2 = plt.subplot(gs[7, :])
data1 = pd.melt(df,
             id_vars=['Patient Gender'],
             value_vars = list(pathology_list),
             var_name = 'Category',
             value_name = 'Count')
data1 = data1.loc[data1.Count>0]
g=sns.countplot(y='Category',hue='Patient Gender',data=data1, palette="Set1",     ax=ax1, order = data1['Category'].value_counts().index)
ax1.set( ylabel="",xlabel="")
ax1.legend(fontsize=20)
ax1.set_title('X Ray Partition- Male/Female (Total Patient Count = 121120)',fontsize=18);

df['Nothing']=df['Finding Labels'].apply(lambda x: 1 if 'No Finding' in x else 0)

data2 = pd.melt(df,
             id_vars=['Patient Gender'],
             value_vars = list(['Nothing']),
             var_name = 'Category',
             value_name = 'Count')
data2 = data2.loc[data2.Count>0]
g=sns.countplot(y='Category',hue='Patient Gender',data=data2,ax=ax2,palette="Set1")
ax2.set( ylabel="",xlabel="No Findings")
ax2.legend('')
plt.subplots_adjust(hspace=.5)

plt.figure(figsize=(15,10))
gs = gridspec.GridSpec(8,1)
ax1 = plt.subplot(gs[:7, :])
ax2 = plt.subplot(gs[7, :])
data1 = pd.melt(df,
             id_vars=['Patient Gender'],
             value_vars = list(pathology_list),
             var_name = 'Category',
             value_name = 'Count')
data1 = data1.loc[data1.Count>0]
g=sns.countplot(y='Category',data=data1,      ax=ax1, order = data1['Category'].value_counts().index)
ax1.set( ylabel="",xlabel="")
ax1.legend(fontsize=20)
ax1.set_title('X Ray Data Set Partition (Total Patient Count = 121120)',fontsize=18);

df['No Finding']=df['Finding Labels'].apply(lambda x: 1 if 'No Finding' in x else 0)

data2 = pd.melt(df,
             id_vars=['Patient Gender'],
             value_vars = list(['No Finding']),
             var_name = 'Category',
             value_name = 'Count')
data2 = data2.loc[data2.Count>0]
g=sns.countplot(y='Category',data=data2,ax=ax2)
ax2.set( ylabel="",xlabel="No Findings")
ax2.legend('')
plt.subplots_adjust(hspace=.5)



f, axarr1 = plt.subplots(7, 2, sharex=True,figsize=(15, 20))
ax2 = plt.subplot(sharex=True,figsize=(15, 20))
i=0
j=0
x=np.arange(0,100,10)
for pathology in pathology_list :
    g=sns.countplot(x='Age',data=df[df['Finding Labels']==pathology], ax=axarr1[i, j],palette="Set1")
    axarr1[i, j].set_title(pathology)
    g.set(xlabel='Age', ylabel='Patient Count')
    g.set_xlim(0,90)
    g.set_xticks(x)
    g.set_xticklabels(x)
    j=(j+1)%2
    if j==0:
        i=(i+1)%7
f.subplots_adjust(hspace=0.3)

f, axarr = plt.subplots(1, 1, sharex=True,figsize=(15, 10))
g=sns.countplot(x='Age',data=df)
g.set(xlabel='Age', ylabel='Patient Count')
g.set_xlim(0,90)
g.set_xticks(x)
g.set_xticklabels(x)
f.subplots_adjust(hspace=0.3)

f, (ax1,ax2) = plt.subplots( 2, figsize=(15, 10))

data = df[df['Follow-up #']<=10]
g = sns.countplot(x='Follow-up #',data=data,palette="Set1",ax=ax1);
g.set(xlabel='Number of followups', ylabel='Patient Count')

ax1.set_title('Follow-up distribution <= 10');
data = df[df['Follow-up #']>10]
g = sns.countplot(x='Follow-up #',data=data,palette="Set1",ax=ax2);
ax2.set_title('Follow-up distribution > 10');

x=np.arange(15,100,10)
g.set(xlabel='Number of followups', ylabel='Patient Count')
g.set_ylim(15,550)
g.set_xlim(15,100)
g.set_xticks(x)
g.set_xticklabels(x)

f.subplots_adjust(top=1)

f, (ax1,ax2) = plt.subplots( 2, figsize=(15, 10))

data = df[df['Follow-up #']<10]
#g = sns.countplot(x='Follow-up #',data=data,palette="Set2",ax=ax1);

ax1.set_title('Follow-up distribution');
data = df[df['Follow-up #']>=10]
g = sns.countplot(x='Follow-up #',data=data,palette="Set1",ax=ax2);
x=np.arange(15,100,10)
g.set_ylim(15,450)
g.set_xlim(15,100)
g.set_xticks(x)
g.set_xticklabels(x)
f.subplots_adjust(top=1)

data=df.groupby('Finding Labels').count().sort_values('Patient ID',ascending=False)
df1=data[['|' in index for index in data.index]].copy()
df2=data[['|' not in index for index in data.index]]
df2=df2[['No Finding' not in index for index in df2.index]]
df2['Finding Labels']=df2.index.values
df1['Finding Labels']=df1.index.values

f, ax = plt.subplots(sharex=True,figsize=(15, 10))
g=sns.countplot(y='Category',data=data1, ax=ax, order = data1['Category'].value_counts().index,color='b',label="Multiple Pathologies")
sns.set_color_codes("muted")
g=sns.barplot(x='Patient ID',y='Finding Labels',data=df2, ax=ax, color="r",label="Single Pathology")
ax.legend(ncol=2, loc="center right", frameon=True,fontsize=20)
ax.set( ylabel="",xlabel="Number of Patients")
ax.set_title("Comparaison between Single or Multiple Pathologies",fontsize=20)      
sns.despine(left=True)

def label_standard_array(labels):
    standard_class=np.zeros(shape=num_classes)
    for i in labels:
       standard_class[i]=1
    return standard_class
    

#def split TODO:training and test validations
class XrayFolderDS(Dataset):
    def __init__(self, root, transform = None):
        self.data = glob.glob(os.path.join(root, IMG_PATTERN))
        self.transform = transform
        # transforms.ToPILImage(),
        self.to_tensor = transforms.Compose([transforms.ToTensor()])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        entry = self.data[index]
        name = os.path.split(entry)[-1]
        # img = cv2.imread(entry)
        img = Image.open(entry,)
        img.load()
        img = self.transform(img)
        labels_name=label_standard_array(labels[name])
        return img,labels_name
        
test_d = XrayFolderDS(TRAIN_DS_LOC,data_transforms['train'])[1][0]
# convert to np and remove extra dim
numpy_d = test_d.numpy()[0] 
print(test_d.size())
plt.imshow(numpy_d, cmap='gray') # Remove gray for green normalized image :D
plt.show()

#Test the loading code
class XrayFolderDS2(Dataset):
    def __init__(self, root, transform = None):
        self.data = glob.glob(os.path.join(root, IMG_PATTERN))
        self.transform = transform
        # transforms.ToPILImage(),
        self.to_tensor = transforms.Compose([transforms.ToTensor()])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        entry = self.data[index]
        name = os.path.split(entry)[-1]
        # img = cv2.imread(entry)
        img = Image.open(entry,)
        img.load()
        img = self.transform(img)
        labels_name=label_standard_array(labels[name])
        return img,labels_name

import line_profiler

def ftst():
    
    ds = DataLoader(tds, batch_size=25, shuffle=True)
    s = 0
    t = time.time()
    for e, i in enumerate(ds):
        s+=len(i)
        if e == 100:
            print(i[0].size())
            break
    print('took {:.4f} seconds '.format(time.time() - t,))

tds = XrayFolderDS2(TRAIN_DS_LOC,data_transforms['train'])

lpx = line_profiler.LineProfiler(tds.__getitem__)
#print(dir(lpx))
lpx(ftst)()
lpx.print_stats()
    

