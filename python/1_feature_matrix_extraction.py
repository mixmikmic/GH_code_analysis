import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.01
set_session(tf.Session(config=config))

from keras.models import load_model
PATH_MODEL = "../Models/LUNA_model_v2_2.h5"
model = load_model(PATH_MODEL)

import numpy as np
import pandas as pd
import os
import time

PATH_VOXELS = '../../data/stage1_voxels_mask/'
VOXEL_SIZE = 64

def feature_vect(patient):     
    import time
    patient_array = np.load(PATH_VOXELS + patient)
    voxels = patient_array['vox']  
    get_ipython().magic('time np.array(model_v24.predict(x= voxels))')
    preds = np.array(model_v24.predict(x= voxels))
    ixs = np.argmax(preds[0])
    
    xmax_malig = np.max(preds[0], axis=0)
    xmax_spiculation = np.max(preds[1], axis=0)
    xmax_lobulation = np.max(preds[2], axis=0)
    xmax_diameter = np.max(preds[3], axis=0)
    
    xsd_malig = np.std(preds[0], axis=0)
    xsd_spiculation = np.std(preds[1], axis=0)
    xsd_lobulation = np.std(preds[2], axis=0)
    xsd_diameter = np.std(preds[3], axis=0)
    
#     locs = patient_array['locs']
    centroids = patient_array['cents']
    shape = patient_array['shape']
    normalized_locs = centroids.astype('float32') / shape.astype('float32')
    
    feats = (np.concatenate([xmax_malig,xmax_spiculation,xmax_lobulation,xmax_diameter,               xsd_malig,xsd_spiculation,xmax_lobulation,xsd_diameter,               normalized_locs[ixs],normalized_locs.std(axis=0)]))        
    return feats

# unit test
start = time.time()
feats = feature_vect(patient = '008464bb8521d09a42985dd8add3d0d2.npz')
print ("It took %.2d s" %(time.time()-start))
feats

get_ipython().magic('time 2+2')





