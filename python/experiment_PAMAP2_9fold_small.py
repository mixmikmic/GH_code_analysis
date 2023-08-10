import sys
import os
sys.path.insert(0, os.path.abspath('../..'))
import numpy as np
import pandas as pd
# mcfly
from mcfly import tutorial_pamap2, modelgen, find_architecture, storage
# Keras module is use for the deep learning
import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution1D, Flatten, MaxPooling1D
from keras.optimizers import Adam
# We can set some backend options to avoid NaNs
from keras import backend as K

datapath = '/media/sf_VBox_Shared/timeseries/PAMAP2/PAMAP2_Dataset/cleaned_12activities_9vars/'
Xs = []
ys = []

ext = '.npy'
for i in range(9):
    Xs.append(np.load(datapath+'X_'+str(i)+ext))
    ys.append(np.load(datapath+'y_'+str(i)+ext))

# Define directory where the results, e.g. json file, will be stored
resultpath = '/media/sf_VBox_Shared/timeseries/PAMAP2/PAMAP2_Dataset/results_tutorial/' 

modelname = 'my_bestmodel'
model_reloaded = storage.loadmodel(resultpath,modelname)

def split_train_test(X_list, y_list, j):
    X_train = np.concatenate(X_list[0:j]+X_list[j+1:])
    X_test = X_list[j]
    y_train = np.concatenate(y_list[0:j]+y_list[j+1:])
    y_test = y_list[j]
    return X_train, y_train, X_test, y_test

def split_train_small_val(X_list, y_list, j, trainsize=500, valsize=500):
    X = np.concatenate(X_list[0:j]+X_list[j+1:])
    y = np.concatenate(y_list[0:j]+y_list[j+1:])
    rand_ind = np.random.choice(X.shape[0], trainsize+valsize, replace=False)
    X_train = X[rand_ind[:trainsize]]
    y_train = y[rand_ind[:trainsize]]
    X_val = X[rand_ind[trainsize:]]
    y_val = y[rand_ind[trainsize:]]
    return X_train, y_train, X_val, y_val

from keras.models import model_from_json

def get_fresh_copy(model, lr):
    model_json = model.to_json()
    model_copy = model_from_json(model_json)
    model_copy.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr),
                  metrics=['accuracy'])
    #for layer in model_copy.layers:
    #    layer.build(layer.input_shape)
    return model_copy

best_model = model_reloaded

import json
with open(resultpath+'modelcomparison.json', 'r') as outfile:
    model_json = json.load(outfile)

best_params = model_json[0]

nr_epochs = 2

np.random.seed(123)
histories, test_accuracies_list, models = [], [], []
for j in range(len(Xs)):
    X_train, y_train, X_test, y_test = split_train_test(Xs, ys, j)
    model_copy = get_fresh_copy(best_model, best_params['learning_rate'])
    datasize = X_train.shape[0]
    
    history = model_copy.fit(X_train[:datasize,:,:], y_train[:datasize,:],
              nb_epoch=nr_epochs, validation_data=(X_test, y_test))
    
    histories.append(history)
    test_accuracies_list.append(history.history['val_acc'][-1] )
    models.append(model_copy)

print(np.mean(test_accuracies_list))
test_accuracies_list

# Calculate 1-NN for each fold:
nr_epochs = 2

np.random.seed(123)
knn_test_accuracies_list = []
for j in range(len(Xs)):
    print("fold ", j)
    X_train, y_train, X_test, y_test = split_train_test(Xs, ys, j)
    acc = find_architecture.kNN_accuracy(X_train, y_train, X_test, y_test, k=1)
    knn_test_accuracies_list.append(acc )

print(np.mean(knn_test_accuracies_list))
accs_compared = pd.DataFrame({'CNN': test_accuracies_list, 'kNN':knn_test_accuracies_list})
accs_compared

modelname = 'my_bestmodel'

for i, model in enumerate(models):
    storage.savemodel(model,resultpath,modelname+str(i))



