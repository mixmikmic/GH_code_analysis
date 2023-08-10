import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.io import loadmat

data = loadmat("respiratory_variability_spectrogram__dataset.mat")
participants_indices = data["subarray"].ravel()
participants = np.unique(participants_indices)

x = data["train_x"]
x = np.swapaxes(x, 1, 2)
x = np.swapaxes(x, 0, 1)
x = x.reshape(-1, 120, 120, 1)
x = tf.Session().run(tf.image.resize_images(x, [28,28]))
y = data["train_y_binary"]
y = np.hstack([y[0].reshape(-1, 1), y[1].reshape(-1, 1)]) # 1 - No Stress and 2 - Stress

for p in participants:
    indices = np.where(participants_indices == p)
    np.save("".join(["x_",str(p)]), x[indices]) 
    np.save("".join(["y_",str(p)]), y[indices])

from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import Input, Dense, Flatten, Conv2D,  MaxPooling2D
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

n_dim = 28
n_channels = 1
n_classes = 2
l2_rate = 0.0001
learning_rate = 3e-4
epochs = 5
batch_size = 8

precision, recall, fscore = [], [], []
cfm = []
for p in range(len(participants)):
    val_X = np.load("".join(["x_",str(participants[p]), ".npy"]))
    val_Y = np.load("".join(["y_",str(participants[p]), ".npy"]))
    training_participants = np.delete(participants, p)
    tr_X = np.empty((0, n_dim, n_dim, n_channels))
    tr_Y = np.empty((0, n_classes))
    for p in training_participants:
        tr_X = np.vstack([tr_X, np.load("".join(["x_",str(p), ".npy"]))])
        tr_Y = np.vstack([tr_Y, np.load("".join(["y_",str(p), ".npy"]))])
    
    K.clear_session()
    X = Input(shape=(n_dim, n_dim, n_channels), name = "input")
    x = Conv2D(12, kernel_size = 4, 
              strides = 1, 
              activation = "relu", 
              kernel_regularizer=l2(l2_rate),
              name = "conv_1")(X)
    x = MaxPooling2D(pool_size = 2)(x)
    x = Conv2D(24, kernel_size = 4, 
              strides = 1, 
              activation = "relu", 
              kernel_regularizer=l2(l2_rate),
              name = "conv_2")(x)
    x = MaxPooling2D(pool_size = 2)(x)
    x = Flatten()(x)
    x = Dense(512, activation = "relu")(x)
    predictions = Dense(2, activation = "sigmoid")(x)

    model = Model(inputs = X, outputs = predictions)
    model.compile(optimizer = Adam(lr = learning_rate), loss = "binary_crossentropy", metrics = ["accuracy"])
    model.fit(tr_X, tr_Y, epochs = epochs, batch_size = batch_size, shuffle = True, verbose = 0)
    
    val_predictions = model.predict(val_X)
    p, r, f, _ = precision_recall_fscore_support(np.argmax(val_Y, 1), np.argmax(val_predictions, 1), average = "binary")
    fscore.append(f)
    precision.append(p)
    recall.append(r)
    cfm.append(confusion_matrix(np.argmax(val_Y, 1), np.argmax(val_predictions, 1)))
    print(f, " ", p, " ", r)
    
print("Avg F-Score: ", round(np.mean(fscore), 4), " Avg Precision: ", round(np.mean(precision), 4),
     " Avg Recall: ", round(np.mean(recall), 4))



