import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter
from datetime import datetime
import pickle
import copy
import datetime
import sys

import geopandas as gpd
import shapely.geometry
import rasterio
import json
import geopandas as gpd
import geopandas_osm.osm
from descartes import PolygonPatch
import h5py 
from scipy.misc import imresize
import shapely.geometry
import cv2

#!sudo pip install xgboost
#!sudo pip install --upgrade xgboost

#import xgboost

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

output_dir = "/home/ubuntu/data/TX_paired/"

geo_df = pickle.load( open( output_dir+"GeoDataFrame_fine_turked.pickled", "rb" ))
geo_df.rename(columns = {'post-storm_full':'post_resized','pre-storm_full':'pre_resized','post-storm_resized':'post_full','pre-storm_resized':'pre_full'}, inplace=True)
geo_df.set_index("tile_no")
geo_df.head(2)

INPUT_SIZE = 256



#identify a training/testing set

geo_df.head(1)

#only use 'good' and verified files
good_geo_df = geo_df[geo_df.bad_image != True]
good_geo_df = good_geo_df[good_geo_df.verified == True]
good_geo_df = good_geo_df[good_geo_df.tile_no <1000]  #take a smaller subset of data for faster testing
len(good_geo_df)



#note, these are the 256 resized images
image_files = sorted([output_dir+file+'.npy' for file in good_geo_df["post_resized"].values])  
before_image_files = sorted([output_dir+file+'.npy' for file in good_geo_df["pre_resized"].values])  
mask_files = sorted([output_dir+file+'.npy' for file in good_geo_df["DBScan_gauss"].values])


len(image_files),len(mask_files)
#min([len(image_files),len(mask_files)])



def TT_tile_no_split(good_geo_df,split=0.8):
    test_no = np.random.choice(good_geo_df.tile_no.values,int((1-split)*len(good_geo_df)))
    train_no = [num for num in good_geo_df.tile_no.values if num not in test_no]
    return train_no, test_no

train_no, test_no = TT_tile_no_split(good_geo_df,split=0.8)

len(train_no),len(test_no)

#write test and train tile_no lists to disk
f = open('/home/ubuntu/Notebooks/test_train_filelists/xgb_train_tile_no_'+str(datetime.datetime.now())+'.txt', 'w')
for item in train_no:f.write("%s\n" % item)
f.close()

f = open('/home/ubuntu/Notebooks/test_train_filelists/xgb_test_tile_no_'+str(datetime.datetime.now())+'.txt', 'w')
for item in test_no:f.write("%s\n" % item)
f.close()

test_no

output_dir+good_geo_df.post_resized[2]+'.npy'

def gen_train(good_geo_df,train_no):
    indexes = copy.deepcopy(train_no)
    while True:
        np.random.shuffle(indexes)

        for index in indexes:
            Xpost = np.load(output_dir+good_geo_df.post_resized[index]+'.npy')
            Xpre = np.load(output_dir+good_geo_df.pre_resized[index]+'.npy')
            Xpre = Xpre.astype('float32')
            Xpost = Xpost.astype('float32')
            pre_mean = 92.36813   #taken from a central common image
            post_mean = 92.21524   #much closer than expected... are these representative?
            
            Xdiff = Xpost/post_mean - Xpre/pre_mean
            
            Xpost = (Xpost-post_mean)/post_mean  #divide by their respective means (per footprint would be even better)
            Xpre =  (Xpre-pre_mean)/pre_mean
            
            R,G,B = Xpost[:,:,0],Xpost[:,:,1],Xpost[:,:,2]
            Xratios_post = np.stack([R/G,R/B,G/B,R/(G+B),G/(R+B),B/(R+G)],axis=2)
            
            R,G,B = Xpre[:,:,0],Xpre[:,:,1],Xpre[:,:,2]
            Xratios_pre  = np.stack([R/G,R/B,G/B,R/(G+B),G/(R+B),B/(R+G)],axis=2)
            
            X = np.concatenate([Xpost,Xdiff,Xpre,Xratios_post,Xratios_pre],axis=2)
             
            Y = np.load(output_dir+good_geo_df.DBScan_gauss[index]+'.npy')
            Y = Y.astype('float32') #/ 255.
            #add extra first dimension for tensorflow compatability
            X = np.expand_dims(X,axis=0)
            Y = np.expand_dims(Y,axis=0)
            Y = np.expand_dims(Y,axis=3)
            yield (X,Y)

def gen_test(good_geo_df,test_no):
    indexes = copy.deepcopy(test_no)
    while True:
        np.random.shuffle(indexes)

        for index in indexes:
            Xpost = np.load(output_dir+good_geo_df.post_resized[index]+'.npy')
            Xpre = np.load(output_dir+good_geo_df.pre_resized[index]+'.npy')
            Xpre = Xpre.astype('float32')
            Xpost = Xpost.astype('float32')
            pre_mean = 92.36813   #taken from a central common image
            post_mean = 92.21524   #much closer than expected... are these representative?
            
            Xdiff = Xpost/post_mean - Xpre/pre_mean
            
            Xpost = (Xpost-post_mean)/post_mean  #divide by their respective means (per footprint would be even better)
            Xpre =  (Xpre-pre_mean)/pre_mean
            
            R,G,B = Xpost[:,:,0],Xpost[:,:,1],Xpost[:,:,2]
            Xratios_post = np.stack([R/G,R/B,G/B,R/(G+B),G/(R+B),B/(R+G)],axis=2)
            
            R,G,B = Xpre[:,:,0],Xpre[:,:,1],Xpre[:,:,2]
            Xratios_pre  = np.stack([R/G,R/B,G/B,R/(G+B),G/(R+B),B/(R+G)],axis=2)
            
            X = np.concatenate([Xpost,Xdiff,Xpre,Xratios_post,Xratios_pre],axis=2)
             
            Y = np.load(output_dir+good_geo_df.DBScan_gauss[index]+'.npy')
            Y = Y.astype('float32') #/ 255.
            #add extra first dimension for tensorflow compatability
            X = np.expand_dims(X,axis=0)
            Y = np.expand_dims(Y,axis=0)
            Y = np.expand_dims(Y,axis=3)
            yield (X,Y)



def get_train(good_geo_df,train_no):
    indexes = copy.deepcopy(train_no)
    
    np.random.shuffle(indexes)
    X_train = []
    Y_train = []
    
    for index in indexes:
        Xpost = np.load(output_dir+good_geo_df.post_resized[index]+'.npy')
        Xpre = np.load(output_dir+good_geo_df.pre_resized[index]+'.npy')
        Xpre = Xpre.astype('float32')
        Xpost = Xpost.astype('float32')
        pre_mean = 92.36813   #taken from a central common image
        post_mean = 92.21524   #much closer than expected... are these representative?

        Xdiff = Xpost/post_mean - Xpre/pre_mean

        Xpost = (Xpost-post_mean)/post_mean  #divide by their respective means (per footprint would be even better)
        Xpre =  (Xpre-pre_mean)/pre_mean

        R,G,B = Xpost[:,:,0],Xpost[:,:,1],Xpost[:,:,2]
        Xratios_post = np.stack([R/G,R/B,G/B,R/(G+B),G/(R+B),B/(R+G)],axis=2)

        R,G,B = Xpre[:,:,0],Xpre[:,:,1],Xpre[:,:,2]
        Xratios_pre  = np.stack([R/G,R/B,G/B,R/(G+B),G/(R+B),B/(R+G)],axis=2)

        X = np.concatenate([Xpost,Xdiff,Xpre,Xratios_post,Xratios_pre],axis=2)

        Y = np.load(output_dir+good_geo_df.DBScan_gauss[index]+'.npy')
        Y = Y.astype('float32') #/ 255.
        
        X = np.reshape(X,(X.shape[0]**2,X.shape[2]))
        Y = np.reshape(Y,(Y.shape[0]**2))
        
        X_train.append(X)
        Y_train.append(Y)
        
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)

    X_train = np.reshape(X_train,(X_train.shape[0]*X_train.shape[1],X_train.shape[2]))
    Y_train = np.reshape(Y_train,(Y_train.shape[0]*Y_train.shape[1]))
                         
    return X_train, Y_train

def get_test(good_geo_df,test_no):
    indexes = copy.deepcopy(test_no)
    
    np.random.shuffle(indexes)
    X_test = []
    Y_test = []
    
    for index in indexes:
        Xpost = np.load(output_dir+good_geo_df.post_resized[index]+'.npy')
        Xpre = np.load(output_dir+good_geo_df.pre_resized[index]+'.npy')
        Xpre = Xpre.astype('float32')
        Xpost = Xpost.astype('float32')
        pre_mean = 92.36813   #taken from a central common image
        post_mean = 92.21524   #much closer than expected... are these representative?

        Xdiff = Xpost/post_mean - Xpre/pre_mean

        Xpost = (Xpost-post_mean)/post_mean  #divide by their respective means (per footprint would be even better)
        Xpre =  (Xpre-pre_mean)/pre_mean

        R,G,B = Xpost[:,:,0],Xpost[:,:,1],Xpost[:,:,2]
        Xratios_post = np.stack([R/G,R/B,G/B,R/(G+B),G/(R+B),B/(R+G)],axis=2)

        R,G,B = Xpre[:,:,0],Xpre[:,:,1],Xpre[:,:,2]
        Xratios_pre  = np.stack([R/G,R/B,G/B,R/(G+B),G/(R+B),B/(R+G)],axis=2)

        X = np.concatenate([Xpost,Xdiff,Xpre,Xratios_post,Xratios_pre],axis=2)

        Y = np.load(output_dir+good_geo_df.DBScan_gauss[index]+'.npy')
        Y = Y.astype('float32') #/ 255.
        
        X = np.reshape(X,(X.shape[0]**2,X.shape[2]))
        Y = np.reshape(Y,(Y.shape[0]**2))
        
        X_test.append(X)
        Y_test.append(Y)
        
    X_test = np.asarray(X_test)
    Y_test = np.asarray(Y_test)

    X_test = np.reshape(X_test,(X_test.shape[0]*X_test.shape[1],X_test.shape[2]))
    Y_test = np.reshape(Y_test,(Y_test.shape[0]*Y_test.shape[1]))
                         
    return X_test, Y_test

X_train,Y_train = get_train(good_geo_df,train_no)

X_test, Y_test  = get_test(good_geo_df,test_no)

X_train.shape, Y_train.shape

len(X_train),X_train[0].shape,Y_train[0].shape

len(X_test),X_test[0].shape,Y_test[0].shape

len(train_no),len(test_no)

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

X_train.shape

X_train.nbytes/1e9,X_test.nbytes/1e9,"in Gigabytes"

model = XGBClassifier(max_depth=9, learning_rate=0.1, n_estimators=10)
#model.fit(X_train, Y_train)
model.fit(X_train, Y_train)

# see how it did

y_pred = model.predict(X_test)

accuracy = (y_pred==Y_test).sum()*1.0/len(y_pred)
accuracy
#default yielded accuracy = 0.747

"""md=9
lr=0.1
n_est=10
print(n_est)
for x in n_est:
    model = XGBClassifier(max_depth=md, learning_rate=lr, n_estimators=x, silent=False,)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    accuracy = (y_pred==Y_test).sum()*1.0/len(y_pred)
    print("x = ",x,"accuracy =",accuracy)"""



index = np.random.choice(test_no,1)[0]
print(index)
#process the image as the training set was
Xview = np.load(output_dir+good_geo_df.post_resized[index]+'.npy')
Xpost = np.load(output_dir+good_geo_df.post_resized[index]+'.npy')
Xpre  = np.load(output_dir+good_geo_df.pre_resized[index]+'.npy')
Xpre  = Xpre.astype('float32')
Xpost = Xpost.astype('float32')
pre_mean = 92.36813   #taken from a central common image
post_mean = 92.21524   #much closer than expected... are these representative?

Xdiff = Xpost/post_mean - Xpre/pre_mean

Xpost = (Xpost-post_mean)/post_mean  #divide by their respective means (per footprint would be even better)
Xpre =  (Xpre-pre_mean)/pre_mean

R,G,B = Xpost[:,:,0],Xpost[:,:,1],Xpost[:,:,2]
Xratios_post = np.stack([R/G,R/B,G/B,R/(G+B),G/(R+B),B/(R+G)],axis=2)

R,G,B = Xpre[:,:,0],Xpre[:,:,1],Xpre[:,:,2]
Xratios_pre  = np.stack([R/G,R/B,G/B,R/(G+B),G/(R+B),B/(R+G)],axis=2)

X = np.concatenate([Xpost,Xdiff,Xpre,Xratios_post,Xratios_pre],axis=2)
Y = np.load(output_dir+good_geo_df.DBScan_gauss[index]+'.npy')
Y = Y.astype('float32') #/ 255.

X = np.reshape(X,(X.shape[0]**2,X.shape[2]))
Y = np.reshape(Y,(Y.shape[0]**2))

plt.imshow(Xview)
plt.show();





prediction = model.predict(X)

predicted_image = np.reshape(prediction,(256,256))

plt.imshow(predicted_image)
plt.show()

plt.imshow(Xview)
plt.imshow(255*predicted_image,alpha=0.3)
plt.show()



prob = model.predict_proba(X)
prob_image = np.reshape(prob[:,1],(256,256))

plt.imshow(prob_image)
plt.show()

plt.imshow(prob_image>0.35)
plt.show()

per_flooded = Y_train.sum()/len(Y_train)
per_flooded

from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

Y_test.shape,y_pred.shape

print(classification_report(Y_test, y_pred))

fpr, tpr, thresholds = roc_curve(Y_test, y_proba)
roc_auc = auc(fpr, tpr)

print('XGBoost model')
print('AUC: ' + str(roc_auc))
print("")

plt.plot(fpr, tpr, color = 'red')
plt.plot([0,1],[0,1], '-.', color = 'grey')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost ROC');
plt.show()





#footprint = "HOLDOUT/2131133.tif"
footprint = "HOLDOUT/2131131.tif"

#load a hold-out footprint:
src_pre = rasterio.open('/home/ubuntu/data/TX_pre/'+footprint)
img_pre = src_pre.read([1, 2, 3]).transpose([1,2,0])
# Plot it
#fig, ax = plt.subplots(figsize=(10,10))
#plt.imshow(img);

src_post = rasterio.open('/home/ubuntu/data/TX_post/'+footprint)
img_post = src_post.read([1, 2, 3]).transpose([1,2,0])

img_post.shape

meta = src_post.meta

str(meta)

#resize them by factor of 2
resize_width = img_post.shape[0]/2
r = resize_width / (1.0*img_post.shape[1])
dim = (resize_width, int(img_post.shape[0] * r))
img_post_resized = cv2.resize(img_post, dim, interpolation = cv2.INTER_AREA)
img_pre_resized  = cv2.resize(img_pre, dim, interpolation = cv2.INTER_AREA)
img_post_resized.shape

plt.figure(figsize=(20,20))
plt.imshow(img_post_resized)
plt.show()

plt.figure(figsize=(20,20))
plt.imshow(img_pre_resized)
plt.show()

#rename image data, so I can use previously written code:
Xpost = img_post_resized
Xpre  = img_pre_resized

#same code as for 1 tile:

pre_mean = 92.36813   #taken from a central common image
post_mean = 92.21524   #much closer than expected... are these representative?

Xdiff = Xpost/post_mean - Xpre/pre_mean

Xpost = (Xpost-post_mean)/post_mean  #divide by their respective means (per footprint would be even better)
Xpre =  (Xpre-pre_mean)/pre_mean

R,G,B = Xpost[:,:,0],Xpost[:,:,1],Xpost[:,:,2]
Xratios_post = np.stack([R/G,R/B,G/B,R/(G+B),G/(R+B),B/(R+G)],axis=2)

R,G,B = Xpre[:,:,0],Xpre[:,:,1],Xpre[:,:,2]
Xratios_pre  = np.stack([R/G,R/B,G/B,R/(G+B),G/(R+B),B/(R+G)],axis=2)

X = np.concatenate([Xpost,Xdiff,Xpre,Xratios_post,Xratios_pre],axis=2)
Y = np.load(output_dir+good_geo_df.DBScan_gauss[index]+'.npy')
Y = Y.astype('float32') #/ 255.

X = np.reshape(X,(X.shape[0]**2,X.shape[2]))
Y = np.reshape(Y,(Y.shape[0]**2))

X.shape



prediction = model.predict(X)

predicted_image = np.reshape(prediction,img_post_resized.shape[:2])

plt.figure(figsize=(20,20))
plt.imshow(predicted_image)
plt.show()

plt.figure(figsize=(20,20))
plt.imshow(img_post_resized)
plt.imshow(255*predicted_image,alpha=0.3)
plt.show()

prob = model.predict_proba(X)
prob_image = np.reshape(prob[:,1],img_post_resized.shape[:2])

plt.figure(figsize=(20,20))
plt.imshow(prob_image)
plt.show()

plt.figure(figsize=(20,20))
plt.imshow(prob_image>0.35)
plt.show()











