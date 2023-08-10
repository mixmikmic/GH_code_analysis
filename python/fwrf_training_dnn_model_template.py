import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu0') # some other variable would need to be set with the gpuarray backend
#import theano.gpuarray
#theano.gpuarray.use('cuda1')

import sys
import os
import struct
import time
import numpy as np
import h5py
from scipy.stats import pearsonr
from tqdm import tqdm
import pickle
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
get_ipython().magic('matplotlib inline')

import src.numpy_utility as pnu
import src.fwrf as fwrf
from src.fwrf import fpX
from src.plots import display_candidate_loss
from src.load_data import load_stimuli, load_voxels

timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())

root_dir   = os.getcwd() + '/'
output_dir = root_dir+"output/"

print "Time Stamp: %s" % timestamp

dataset_dir = "/home/styvesg/Documents/PostDoc/Datasets/vim-1/"

stimuli_lowrez, stimuli_hirez, trn_size = load_stimuli(dataset_dir, npx=227, npc=3)

data_size = len(stimuli_hirez)
val_size = data_size - trn_size

trn_stim_data = stimuli_hirez[:trn_size]
val_stim_data = stimuli_hirez[trn_size:]

plt.imshow(trn_stim_data[5].transpose((1,2,0)))

subject = 'S1'
roi_names = ['other', 'V1', 'V2', 'V3', 'V3a', 'V3b', 'V4', 'LO']

voxel_data, voxel_roi, voxel_idx = load_voxels(dataset_dir, subject, voxel_subset=range(3400, 3700))

nv = voxel_data.shape[1]
print "nv = %d" % nv

trn_voxel_data = voxel_data[:trn_size]
val_voxel_data = voxel_data[trn_size:]

model_name = 'deepnet'

trn_feature_dict = h5py.File(dataset_dir + "caffe_refnet_trn_response.h5py", 'r')   # 'r' means that hdf5 file is open in read-only mode
val_feature_dict = h5py.File(dataset_dir + "caffe_refnet_val_response.h5py", 'r')

layerlist = trn_feature_dict.keys()
print layerlist

# concatenate and sort as list
fmap_max = 1024
order = layerlist[0:8] #+layerlist[7:8]
fmaps = []
fmaps_sizes = []
fmaps_count = 0
for l in order:
    fmap = np.concatenate((np.array(trn_feature_dict[l], dtype=fpX), np.array(val_feature_dict[l], dtype=fpX)), axis=0)        
    if fmap.ndim==2:
        fmap = fmap.reshape(fmap.shape+(1,1))
    if fmap.shape[1]>fmap_max:
        #select the feature map with the most variance to the dataset
        fmap_var = np.var(fmap[:trn_size], axis=(0,2,3))
        most_var = fmap_var.argsort()[-fmap_max:]
        fmap = fmap[:,most_var,:,:]
    print "layer: %s, shape=%s" % (l, (fmap.shape))
    fmaps += [fmap,]
    fmaps_sizes  += [fmap.shape,]
    fmaps_count += fmap.shape[1]
    
trn_feature_dict.close()
val_feature_dict.close()

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(trn_stim_data[5,0,:,:], cmap='gray')
plt.subplot(1,3,2)
plt.imshow(fmaps[0][5,2,:,:], interpolation='None')
plt.colorbar()
plt.subplot(1,3,3)
plt.imshow(fmaps[1][5,10,:,:], interpolation='None')
plt.colorbar()

# aliases
nf = fmaps_count
fmaps_res_count = len(fmaps)

# make sure we are using float32
print fmaps[0].dtype

lx = ly = 20.
nx = ny = 10
smin, smax = 0.7, 8.
ns = 8

# sharedModel specification is a list of 3 ranges and 3 callable functor. The reason for this is for a future implementation of dynamic mesh refinement.
sharedModel_specs = [[(0., lx), (0., ly), (smin, smax)], [fwrf.linspace(nx), fwrf.linspace(ny), fwrf.logspace(ns)]]
# initial values of the fwrf model parameters
voxelParams = [np.full(shape=(nv, nf), fill_value=0.0, dtype=fpX), np.full(shape=(nv), fill_value=0.0, dtype=fpX)]

rx = sharedModel_specs[1][0](*sharedModel_specs[0][0])
ry = sharedModel_specs[1][1](*sharedModel_specs[0][1])
rs = sharedModel_specs[1][2](*sharedModel_specs[0][2])
print "G = %d\n" % (nx*ny*ns)
#print range:
print "range x"
print rx
print "range y"
print ry
print "range s"
print rs

ith_rf_size = 0

n = len(fmaps_sizes)
plt.figure(figsize=(5*n,5))
sigmas = sharedModel_specs[1][2](*sharedModel_specs[0][2])
for i,r in enumerate(fmaps_sizes):
    _,_,z = pnu.make_gaussian_mass(0., 0., sigmas[ith_rf_size], r[2], size=20.)
    plt.subplot(1,n,i+1)
    plt.imshow(z, interpolation='None', cmap='jet')
    plt.colorbar()

log_act_func = lambda x: np.log(1+np.sqrt(np.abs(x)))

mst_data, mst_avg, mst_std = fwrf.model_space_tensor(fmaps, sharedModel_specs, nonlinearity=log_act_func,                 zscore=True, trn_size=trn_size, epsilon=1e-6, batches=(200, nx*ny), view_angle=lx, verbose=True, dry_run=False)
print mst_data.shape

print np.amin(mst_data), np.amax(mst_data)
# split the model space tensor into trn and val set.
trn_mst_data = mst_data[:trn_size]
val_mst_data = mst_data[trn_size:]

val_scores, best_scores, best_epochs, best_candidates, best_w_params = fwrf.learn_params(
        trn_mst_data, trn_voxel_data, voxelParams, batches=(200, 300, nx*ny), \
        val_test_size=350, lr=1e-4, l2=0.0, num_epochs=40, output_val_scores=-1, output_val_every=1, verbose=True, dry_run=False)

best_rf_params, best_avg, best_std = fwrf.real_space_model(best_candidates, sharedModel_specs, mst_avg=mst_avg, mst_std=mst_std)

val_pred, val_cc = fwrf.get_prediction(val_mst_data, val_voxel_data, best_candidates, best_w_params, batches=(200, 10*ny*nx))

print "max cc = %f" % np.max(val_cc)
print "sum(cc>0.2) = %d" % np.sum(map(lambda x: x > 0.2, val_cc))
plt.figure(figsize=(10,5))
_=plt.hist(val_cc[:], bins=100, range=(-.5, 1.))
plt.yscale('log')
plt.ylim([10**-1, 10**3])
plt.xlim([-.4, 0.9])

_=plt.hist(best_epochs, bins=40)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
_,_,g_stack = pnu.make_gaussian_stack(best_rf_params[:,0], best_rf_params[:,1], best_rf_params[:,2], 64, size=20., dtype=fpX)
plt.imshow(np.sum(g_stack, axis=0), cmap='magma', interpolation='None')
plt.subplot(1,2,2)
_=plt.hist(best_rf_params[:,2], bins=50)
plt.yscale('log')

vidx = np.argmax
print best_rf_params[vidx,:]
fig1 = display_candidate_loss(val_scores[-1,vidx,:], nx, ny, ns)

for v in range(val_scores.shape[1]):
    plt.plot(val_scores[:,v,best_candidates[v]])

plt.plot(val_pred[:,vidx])
plt.plot(val_voxel_data[:,vidx])

print np.corrcoef(val_pred[:,vidx], val_voxel_data[:,vidx])

ex_file_name = "fwrf_%s_%s_%s_data.pkl" % (model_name, subject, timestamp)
ex_file = open(output_dir + ex_file_name, 'wb')
ex_values = {'project': 'fwrf',
             'dataset': 'vim-1',
             'subject': subject,
             'model_name': model_name,
             'grid': [sharedModel_specs[1][0](*sharedModel_specs[0][0]), 
                      sharedModel_specs[1][1](*sharedModel_specs[0][1]),
                      sharedModel_specs[1][2](*sharedModel_specs[0][2])],
             'fmaps_res_count': fmaps_res_count,
             'fmaps_count': fmaps_count,
             'fmaps_sizes': fmaps_sizes,      
             'scores': best_scores,
             'rf_params': best_rf_params,
             'w_params': best_w_params,
             'normavg': best_avg,
             'normstd': best_std, 
             'val_pred': val_pred,
             'val_cc': val_cc}
pickle.dump(ex_values, ex_file)
ex_file.close()
print ex_file_name

#find the start and end point of the feature map partitions
fmaps_count = len(fmaps_sizes)
partitions = [0,]
for r in fmaps_sizes:
    partitions += [partitions[-1]+r[1],]
print partitions

partition_val_pred = np.ndarray(shape=(fmaps_count,)+val_pred.shape, dtype=fpX)
partition_val_cc   = np.ndarray(shape=(fmaps_count,)+val_cc.shape, dtype=fpX)

for l in range(fmaps_count):
    partition_params = [np.zeros(p.shape, dtype=fpX) for p in best_w_params]  
    partition_params[0][:, partitions[l]:partitions[l+1]] = best_w_params[0][:, partitions[l]:partitions[l+1]]
    partition_params[1][:] = best_w_params[1][:]

    partition_val_pred[l,...], partition_val_cc[l,...] = fwrf.get_prediction(val_mst_data, val_voxel_data, best_candidates, partition_params, batches=(200, 10*ny*nx))

# calculate covariances
partition_r = np.ndarray(shape=(fmaps_count, nv))
for v in range(nv):
    full_c = np.cov(val_pred[:,v], val_voxel_data[:,v])
    for l in range(fmaps_count):
        part_c = np.cov(partition_val_pred[l,:,v], val_voxel_data[:,v])
        partition_r[l,v] = part_c[0,1]/np.sqrt(full_c[0,0]*full_c[1,1])

part_file = open(output_dir + "fwrf_%s_%s_%s_part.pkl" % (model_name, subject, timestamp), 'wb')
part_values = {'dataset': 'vim-1',
             'subject': subject,
             'model_name': model_name,
             'val_pred': partition_val_pred,
             'val_cc': partition_val_cc,
             'val_ri': partition_r}
pickle.dump(part_values, part_file)
part_file.close()

nROI = int(np.max(voxelROI))+1
nL   = fmaps_count

partition_R_avg = np.ndarray(shape=(fmaps_count, nROI), dtype=fpX)
partition_R_std = np.ndarray(shape=(fmaps_count, nROI), dtype=fpX)
for roi in range(nROI):
    roi_mask = np.logical_and(voxelROI.flatten()==roi, val_cc>0.27)    
    for l in range(fmaps_count):
        partition_R_avg[l,roi] = np.mean(partition_r[l, roi_mask] /  val_cc[roi_mask])
        partition_R_std[l,roi] = np.std(partition_r[l, roi_mask])
#plt.imshow(partition_R_avg, interpolation='None')

from matplotlib.pyplot import cm 
color=iter(cm.magma(np.linspace(0,1,nL)))

plt.figure(figsize=(20,10))
c=next(color)
plist = []
_ = plt.bar(np.arange(len(roi_names)), partition_R_avg[0,:], yerr=partition_R_std[0,:], color=c, align='center')
plist += [_,]
for l in range(1,nL):
    c=next(color)
    _= plt.bar(np.arange(len(roi_names)), partition_R_avg[l,:], bottom=np.sum(partition_R_avg[:l,:],axis=0), yerr=partition_R_std[l,:], color=c,        align='center', tick_label=roi_names)
    plist += [_,]
plt.legend(plist, ['layer %d' % l for l in range(1,len(plist)+1)])
plt.ylim([0,1])
plt.ylabel('Layer contribution to total prediction accuracy\n (averaged over voxels in roi)')



