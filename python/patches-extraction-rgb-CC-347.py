get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pylab as plt
import numpy as np
import nibabel as nib
import os
import glob
import sys
from sklearn.feature_extraction.image import extract_patches_2d
import random
import time

MY_UTILS_PATH = "../Modules/"
if not MY_UTILS_PATH in sys.path:
    sys.path.append(MY_UTILS_PATH)
import ipt_utils
import cnn_utils

patch_size = (64,64) 
max_patches = 5 # number of patches per slice
nslices = 120 # number of slices
offset = 20 # Offset to get non-brain slices
t = 0.5 # STAPLE threshold for getting mask bounding-box

#Adapt the following 3 paths accordingly
orig_path = "/media/roberto/DATA/CC-359/Original/Original"
staple_path = "/media/roberto/DATA/CC-359/STAPLE/STAPLE"
patches_path = "/media/roberto/DATA/Patches-CC347"


cc347_list =np.genfromtxt("../Data/cc347_orig.txt", dtype ="string")
cc347_staple = np.genfromtxt("../Data/cc347_staple.txt",dtype = "string")
print cc347_list[0]
print cc347_staple[0]

for (orig,staple_auto) in zip(cc347_list,cc347_staple):
    # Initializing arrays to store the image and label patches
    patches_orig = np.zeros((nslices*max_patches,patch_size[0],patch_size[1],3))
    patches_staple_auto = np.zeros((nslices*max_patches,patch_size[0],patch_size[1]))
    
    # Loading original volume
    orig_data = os.path.join(orig_path,orig)
    orig_data = nib.load(orig_data).get_data()
    
    #Data normalization
    orig_min = orig_data.min()
    orig_max = orig_data.max()
    orig_data = 1000.0*(orig_data - orig_min)/(orig_max-orig_min)
    orig_data = orig_data.astype(np.uint16) # we save as uint16 to save space
        
    #Loading STAPLE mask
    staple_auto = os.path.join(staple_path,staple_auto)
    staple_auto = (nib.load(staple_auto).get_data())
    
    # Get bounding-box of the mask
    H,W,Z = orig_data.shape
    xmin,xmax,ymin,ymax,zmin,zmax = ipt_utils.crop3D(staple_auto > t)
    
    
    # Offsets the bounding-box to get non-brain samples
    xmin,xmax = np.maximum(xmin-offset,0),np.minimum(xmax+offset,H)
    ymin,ymax = np.maximum(ymin-offset,0),np.minimum(ymax+offset,W)
    zmin,zmax = np.maximum(zmin-offset,0),np.minimum(zmax+offset,Z)
    
    # crop volumes
    orig_data = orig_data[xmin:xmax,ymin:ymax,zmin:zmax]
    staple_auto = staple_auto[xmin:xmax,ymin:ymax,zmin:zmax]
    
    H2,W2,Z2 = orig_data.shape
    random.seed(time.time())
    np.random.seed(int(time.time()))
    counter = 0
    for ii in random.sample(range(1, W2-1), nslices):
        aux_orig = orig_data[:,ii-1:ii+2,:]
        staple_man_auto = staple_auto[:,ii,:]
        
        # Random seed for patch extraction
        rs = np.random.randint(0,1001)
        
            
        patches_orig_aux1 = extract_patches_2d(aux_orig[:,0,:],                            patch_size,max_patches,random_state = rs)
        patches_orig_aux2 = extract_patches_2d(aux_orig[:,1,:],                            patch_size,max_patches,random_state = rs)
        patches_orig_aux3 = extract_patches_2d(aux_orig[:,2,:],                            patch_size,max_patches,random_state = rs)
            
        patches_orig_aux = np.concatenate((patches_orig_aux1[:,:,:,np.newaxis],                                           patches_orig_aux2[:,:,:,np.newaxis],
                                           patches_orig_aux3[:,:,:,np.newaxis]),axis = -1)
            
        patch_auto_aux_staple = extract_patches_2d(staple_man_auto,patch_size,max_patches,random_state = rs)
        
        patches_orig[counter*max_patches:(counter+1)*max_patches] = patches_orig_aux
        patches_staple_auto[counter*max_patches:(counter+1)*max_patches] = patch_auto_aux_staple
        counter+=1
        
    file_name = orig.split("/")[-1].split(".")[0]
    
    #save patches
    np.save(os.path.join(patches_path,file_name+"_orig.npy"),patches_orig)
    np.save(os.path.join(patches_path,file_name+"_staple.npy"),patches_staple_auto)
    
    print file_name
    print patches_orig.shape
    print patches_staple_auto.shape

fig, ax = plt.subplots(nrows=5, ncols=5)
counter = 0
indexes = np.random.choice(patches_orig.shape[0],25,replace= False)
for row in ax:
    for col in row:
        col.imshow(patches_orig[indexes[counter],:,:,1], cmap = 'gray')
        col.imshow(patches_staple_auto[indexes[counter],:,:]>0.5, cmap = 'cool',alpha = 0.2)
        col.axis("off")
        counter+=1
plt.show()



