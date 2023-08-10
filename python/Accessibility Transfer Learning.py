from keras.models import model_from_json
import json

# json_path = "../model_files/atac_xferlearn_jul24/record_12_model_FbFup_modelJson.json"
# json_path = "../model_files/sharpr_znormed_jul23/record_13_model_bgGhy_modelJson.json"
# json_path = "../model_files/regressionJun24Positives/record_2_model_Yjv2n_modelJson.json"
json_path = "../model_files/atac_xferlearn_jul24/transferlearn_json_keras1.json"
# json_path = "../model_files/atac_xferlearn_jul24/record_3_model_RatMV_modelJson.json"
with open(json_path) as json_file:
    json_string = json.dumps(json.load(json_file))
    model = model_from_json(json_string) 
    
print(model.summary())

import h5py
import numpy as np

# (300, 4, 17, 1) --> (17, 1, 4, 300)
# swap 0,3; 1,2; 0,1: 0, 1, 2, 3 --> 3, 1, 2, 0 --> 3, 2, 1, 0 --> 2, 3, 1, 0

f = h5py.File('../model_files/atac_xferlearn_jul24/record_12_model_FbFup_modelWeights.h5', 'r')

# Checking to see if shapes match up
# # print f.keys()
# # print model.layers[0].name
# print "Previous model weight shapes"
# for i in range(len(f.keys())):
#     layer = 'layer_%d' % i
# #     print layer
# #     print f[layer].keys()
#     for params in f[layer].keys():
#         print f[layer][params].shape

# print "\nNew model weight shapes"
# for w in model.get_weights():
#     print w.shape

weights = []
for i in range(len(f.keys())):
    layer = 'layer_%d' % i
#     if len(f[layer].keys()) == 4:
#         batchnorm_params = f[layer].keys()
#         print batchnorm_params
# #         print f[layer][batchnorm_params[2]].shape
#         weights.append(f[layer][batchnorm_params[0]])
#         weights.append(f[layer][batchnorm_params[1]])
#         weights.append(f[layer][batchnorm_params[2]])
#         weights.append(f[layer][batchnorm_params[3]])
#         continue
    for params in f[layer].keys():
        layer_W = np.array(f[layer][params])
        if len(layer_W.shape) == 4:
            layer_W = np.swapaxes(np.swapaxes(np.swapaxes(layer_W, 0, 3), 1, 2), 0, 1)
        weights.append(layer_W)

model.set_weights(weights)

import h5py
import numpy as np
import time

train_data = h5py.File("../hdf5files/atac_xferlearn_jul24/train_data.hdf5")
X_train = train_data['X']['sequence']
y_train_pred = np.ndarray(shape = (len(X_train), 16))

batch_size = 500
t0 = time.time()
print "Total batches: %d" % (len(X_train)/batch_size + 1)
for i in range(len(X_train)/batch_size + 1):
    if i % 100 == 0 and i > 0:
        print("Batches %d to %d took %.3f sec" % (i-100, i, time.time() - t0))
        t0 = time.time()
        print "On batch %d" % i
    if (i+1)*batch_size > len(X_train):
        y_train_pred[i*batch_size:] = model.predict_on_batch(X_train[i*batch_size:])
    y_train_pred[i*batch_size : (i+1)*batch_size] = model.predict_on_batch(X_train[i*batch_size : (i+1)*batch_size])

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
import time

def generate_predictions(model, X, ntasks, batchsize=500):
    y_pred = np.ndarray(shape = (len(X), ntasks))
    
    t0 = time.time()
    print "Total batches: %d" % (len(X)/batchsize + 1)
    for i in range(len(X)/batchsize + 1):
        if (i % 100 == 0 or i == len(X)/batchsize) and i > 0:
            print("Batches %d to %d took %.3f sec" % (i-100, i, time.time() - t0))
            t0 = time.time()
            print "On batch %d" % i
        if (i+1)*batchsize > len(X):
            y_pred[i*batchsize:] = model.predict_on_batch(X[i*batchsize:])
        y_pred[i*batchsize : (i+1)*batchsize] = model.predict_on_batch(X[i*batchsize : (i+1)*batchsize])
    
    return y_pred

def evaluate_predictions(y_true, y_pred, ntasks):
    for i in range(ntasks):
        y_true_task = y_true[:, i]
        y_pred_task = y_pred[:, i]
        auroc = roc_auc_score(y_true_task, y_pred_task)
        auprc = average_precision_score(y_true_task, y_pred_task)
        sprmn = spearmanr(y_true_task, y_pred_task)
        print "Task %d: AUROC = %.3f, AUPRC = %.3f, Spearman = %.3f, p = %.3f" % (i, auroc, auprc, sprmn[0], sprmn[1])

val_data = h5py.File("../hdf5files/atac_xferlearn_jul24/valid_data.hdf5")
X_val = val_data['X']['sequence'][:600]

y_pred_val = generate_predictions(model, X_val, batchsize = 500, ntasks = 16)

# y_train_true = train_data['Y']['output']
y_val_true = val_data['Y']['output'][:600]
print np.sum(y_val_true[:, 0])
print np.sum(y_pred_val[:, 0])

# print "Training set evaluation"
# evaluate_predictions(y_train_true, y_train_pred, ntasks=16)
print "\nValidation set evaluation"
evaluate_predictions(y_val_true, y_pred_val, ntasks=16)

# On cpu, ~30s for 15 sample batch. On gpu, 0.5 sec.

import time

t0 = time.time()
y = model.predict_on_batch(X_train[i*batch_size : (i+1)*batch_size])
print(time.time() - t0)

import h5py
import numpy as np

fnames = ['valid_data.hdf5', 'train_data.hdf5', 'test_data.hdf5']
fnames = ['../hdf5files/atac_xferlearn_jul24/pretrain_' + name for name in fnames]
windows_per_seq = 5
seqlen = 145
start_indices = np.arange(500 - seqlen*windows_per_seq/2, 500 + seqlen*windows_per_seq/2, seqlen)
print start_indices
for fname in fnames:
    print "On file %s" % fname
    f = h5py.File(fname, 'r+')
    
    # Sequences
    sequences = np.array(f['X/sequence'])
    print sequences.shape
    new_sequences = np.ndarray(shape = (windows_per_seq*len(sequences), seqlen, 4))
    for i in range(windows_per_seq):
        new_sequences[np.arange(i, len(new_sequences), windows_per_seq)] = sequences[:, start_indices[i] : start_indices[i] + 145]
    print new_sequences.shape
    del f['X/sequence']
    f.create_dataset('X/sequence', data = new_sequences)
    
    # Labels
    labels = np.array(f['Y/output'])
    print labels.shape
    new_labels = np.repeat(labels, windows_per_seq, axis=0)
    print new_labels.shape
    del f['Y/output']
    f.create_dataset('Y/output', data = new_labels)
    
    f.close()
    

f = h5py.File('../hdf5files/atac_xferlearn_jul24/pretrain_valid_data.hdf5')
print f['X/sequence'].shape
print f['Y/output'].shape
print f['Y/output'][5:10]

import keras
from keras.layers.core import Dense
from keras.models import model_from_json, Sequential
import numpy as np
import h5py

atac_json = ("../model_files/atac_xferlearn_jul24/record_3_model_6nOFH_modelJson.json")
atac_h5 = ("../model_files/atac_xferlearn_jul24/record_3_model_6nOFH_modelWeights.h5")

model = model_from_json(open(atac_json).read())
model.load_weights(atac_h5)

print model.layers[-1] # sanity check that this is the Dense layer

mpra_layers = model.layers[:-1]
output_layer = Dense(output_dim = 12, 
                     activation = 'linear', 
                     init = 'glorot_uniform')
# output_layer = Dense(12)
mpra_layers.append(output_layer)

mpra_model = Sequential(mpra_layers)

print(mpra_model.summary())

mpra_json = mpra_model.to_json()
with open("../model_files/atac_xferlearn_jul24/atacPretrainedJson.json", "w") as json_file:
    json_file.write(mpra_json)

mpra_model.save_weights("../model_files/atac_xferlearn_jul24/atacPretrainedWeights.h5")

from keras.models import model_from_json
import json

# Load architecture from any one of the many JSON files
json_path = "../model_files/sharpr_znormed_jul23/record_13_model_bgGhy_modelJson.json"
with open(json_path) as json_file:
    json_string = json.dumps(json.load(json_file))
    model = model_from_json(json_string) 
    
# print(model.summary())

weights_files = ["../model_files/sharpr_znormed_jul23//record_13_model_bgGhy_modelWeights.h5", # 0.191
                 "../model_files/sharpr_znormed_jul23//record_14_model_10Sx5_modelWeights.h5", # 0.186
                 
                ]

