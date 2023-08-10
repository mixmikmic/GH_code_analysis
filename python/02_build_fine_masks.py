import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import Counter
import datetime
import pickle
import copy

import os
import rasterio
import shapely.geometry

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

from sklearn.cluster import dbscan

output_dir = "/home/ubuntu/data/TX_paired/"

#first time use:
geo_df = pickle.load( open( output_dir+"GeoDataFrame.pickled", "rb" ))

#after use:
#geo_df = pickle.load( open( output_dir+"GeoDataFrame_fine.pickled", "rb" ))
geo_df['DBScan']=None
geo_df.set_index("tile_no")
geo_df.head(1)



#get tile_no for those tiles with more than a little flooding
t = geo_df["%flooded"] > 0.00
sum(t)
tiles = geo_df[t].tile_no

tiles.values

#run a subset of data at a time in case DBSCAN kills the kernel
for tile_no in tiles.values[:]:
    print("working on tile #:",tile_no)
    
    #load files
    img_post = np.load(output_dir+'%d_post_resize_img.npy'%tile_no)
    img_pre  = np.load(output_dir+'%d_pre_resize_img.npy'%tile_no)
    mask = np.load(output_dir+'%d_resize_mask.npy'%tile_no)
    
    #combine features
    features=img_post
    #add in subtracted image is a feature
    #img_diff = 0.2*(img_post-img_pre)
    #features = np.stack((img_post[:,:,0],img_post[:,:,1],img_post[:,:,2],img_diff[:,:,0],img_diff[:,:,1],img_diff[:,:,2]),axis=2)
    #features.shape
    
    flat_img = np.reshape(features,(features.shape[0]**2,features.shape[2]))
    
    clustered = dbscan(flat_img,eps=1.7,min_samples=100,n_jobs=-2)
    
    #make new mask (of all labels for now)
    side = int(clustered[1].shape[0]**0.5)
    DBScan_mask = np.reshape(clustered[1],(side,side))
    #make backup copy to save for review
    DBScan_mask_original = copy.deepcopy(DBScan_mask)
    
    #order the clusters
    c_count=Counter(DBScan_mask.flatten())
    order = [x[0] for x in c_count.most_common() if x[0]>= 0]  #gets cluster index, AND throws out the negative-1 group

    
    #routine that checks of the masked area is too grey or green, and moves down the list of clusters it is
    
    ready = False
    dry = False
    while ready==False:
        if len(order)==0:
            ready = True
            dry = True
            c_id = None
            break
            #continue

        c_id = order[0]
        color_sum = (img_post*(np.expand_dims(mask==c_id,axis=2))).sum(axis=(0,1))
        #print(color_sum)
        #print(1.0*color_sum[1]/color_sum[0])

        #reject if too grey/white/black 0.108-->0.102-->0.100
        if color_sum.std()*1.0/color_sum.mean() < 0.100:   
            print("too grey, reject cluster",order[0],color_sum.std()*1.0/color_sum.mean())
            order.pop(0)
            continue

        #if the most common color is too green, reject it and move to the next  1.1-->1.2-->1.18--> 1.13-->1.1

        elif 1.0*color_sum[1]/color_sum[0] > 1.2:  
            print(color_sum[1]*1.0/color_sum[0])
            print("too green, reject cluster ",order[0])
            order.pop(0)

        else:ready=True

    print("floodwater/mud at id",c_id)
    
    if dry == True: fine_mask = np.zeros((side,side),dtype='int64')  #the mask data type should match
    else: fine_mask = 1*(DBScan_mask==c_id)
    
    np.save(output_dir+"%d_256_DBSCAN"%tile_no, DBScan_mask_original)
    np.save(output_dir+"%d_256_fine_mask"%tile_no, fine_mask)
        
    #update the entry to the geopandas Dataframe with the filename
    geo_df.DBScan[tile_no] = "%d_256_DBSCAN"%tile_no
    geo_df.fine_make_filename[tile_no] = "%d_256_fine_mask"%tile_no

    #write geopandas to file too
    geo_df.to_pickle(output_dir+"GeoDataFrame_fine.pickled")



"""    #plot for monitoring
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(18,6))
    ax1.set_title("Post-flood Image")
    ax2.set_title("Post-flood Image w/ mask")
    ax3.set_title("Pre-flood Image")
    ax4.set_title("clusters")
    ax1.imshow(img_post)
    ax2.imshow(img_post)
    ax2.imshow(fine_mask,cmap='bwr',alpha = 0.2)
    ax3.imshow(img_pre)
    plt.imshow(DBScan_mask)
    plt.colorbar()
    plt.show();"""























