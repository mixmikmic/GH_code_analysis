import os
import sys
import time
import numpy as np
from sklearn.model_selection import train_test_split
from skimage import color

from scipy import misc
import gc

import keras.callbacks as cb
import keras.utils.np_utils as np_utils
from keras import applications
from keras import regularizers
from keras.models import Sequential
from keras.constraints import maxnorm
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dense, Dropout, Flatten, GaussianNoise

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams['figure.figsize'] = (10,10)
np.set_printoptions(precision=2)

import MIAShelper as MIAS

SEED = 7
np.random.seed(SEED)

CURR_DIR  = os.getcwd()
ImagesDir  = 'C:\\Users\\310127474\\DDSMData\\MIAS\\MIAS\\'
AugmentedImagesDir   = 'C:\\Users\\310127474\\DDSMData\\MIAS\\MIAS\\AUG'
meta_file = 'meta_data_mias.csv'
PATHO_INX = 6    # Column number of pathology label in meta_file
FILE_INX  = 1    # Column number of File name in meta_file

meta_data, _ = MIAS.load_meta(meta_file, patho_idx=5, file_idx=1)
meta_data = MIAS.clean_meta(meta_data,ImagesDir)
items = ['normal','abnormal']
labels = {}
for i, item in enumerate(items):
    labels[item] = i

X_data, Y_data = MIAS.load_data(meta_data, ImagesDir, labels,imgResize=(224,224))

fig, axs = plt.subplots(8,8, figsize=(16, 16))
#fig.subplots_adjust(hspace = .2, wspace=.001)
axs = axs.ravel()

# Pick 32 Images for Display
for i in np.arange(64):
    axs[i].axis('off')
    axs[i].set_title('Mammo', fontsize=10)
    axs[i].imshow(X_data[i,:,:],cmap='gray')

datagen = ImageDataGenerator(rotation_range=5, width_shift_range=.01, height_shift_range=0.01,
                             data_format='channels_first')

cls_cnts = MIAS.get_clsCnts(Y_data, labels)
X_data, Y_data = MIAS.balanceViaSmote(cls_cnts, meta_data, ImagesDir, AugmentedImagesDir, labels, 
                                    datagen, X_data, Y_data, imgResize=(224,224),seed=SEED, verbose=True)

X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data,
                                                    test_size=0.25,
                                                    random_state=SEED,
                                                    )
print( X_train.shape)
print( X_test.shape)
print (Y_train.shape)
print (Y_test.shape)

data = [X_train, X_test, Y_train, Y_test]

def VGGPrep(img_data):
    
    images = np.zeros([len(img_data), img_data.shape[1], img_data.shape[2], 3])
    for i in range(0, len(img_data)):
        im = (img_data[i] * 255)        # Original imagenet images were not rescaled
        im = color.gray2rgb(im)
        images[i] = im
    return(images)

def vgg16_bottleneck(data, modelPath, fn_train_feats, fn_train_lbls, fn_test_feats, fn_test_lbls):
    # Loading data
    X_train, X_test, Y_train, Y_test = data
    
    X_train = VGGPrep(X_train)
    X_test = VGGPrep(X_test)
        
    model = applications.VGG16(include_top=False, weights='imagenet') 
    
    # Predict passes data through the model 
    bottleneck_features_train = model.predict(X_train)
    
    # Saving the bottleneck features for the training data
    featuresTrain = os.path.join(modelPath, fn_train_feats)
    labelsTrain = os.path.join(modelPath, fn_train_lbls)
    
    np.save(open(featuresTrain, 'wb'), bottleneck_features_train)
    np.save(open(labelsTrain, 'wb'), Y_train)

    
    bottleneck_features_test = model.predict(X_test)
    
    # Saving the bottleneck features for the test data
    featuresTest = os.path.join(modelPath, fn_test_feats)
    labelsTest = os.path.join(modelPath, fn_test_lbls)
    np.save(open(featuresTest, 'wb'), bottleneck_features_test)
    np.save(open(labelsTest, 'wb'), Y_test)

# Locations for the bottleneck and labels files that we need
train_bottleneck = 'features_train.npy'
train_labels     = 'labels_train.npy'
test_bottleneck  = 'features_test.npy'
test_labels      = 'labels_test.npy'
modelPath = os.getcwd()

top_model_weights_path = './weights/'

np.random.seed(SEED)
vgg16_bottleneck(data, modelPath, train_bottleneck, train_labels, test_bottleneck, test_labels)

class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []
        
    def on_epoch_end(self, epoch, logs={}):
        epoch_tr_loss  = logs.get('loss')
        epoch_val_loss = logs.get('val_loss')
        self.losses.append([epoch_tr_loss, epoch_val_loss])
        
        epoch_tr_acc  = logs.get('acc')
        epoch_val_acc = logs.get('val_acc')
        self.acc.append([epoch_tr_acc, epoch_val_acc])

def plot_losses(losses, acc):
    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.plot(losses)
    ax.set_title('Model Loss')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'test'], loc='upper left')
    
    ax = fig.add_subplot(222)
    ax.plot(acc)
    ax.set_title('Model Accuracy')
    ax.set_ylabel('accuracy')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'test'], loc='upper left')

def train_model(train_feats, train_lab, test_feats, test_lab, model_path, model_save, epoch = 50, batch = 64):
    start_time = time.time()
    
    train_bottleneck = os.path.join(model_path, train_feats)
    train_labels = os.path.join(model_path, train_lab)
    test_bottleneck = os.path.join(model_path, test_feats)
    test_labels = os.path.join(model_path, test_lab)
    
    history = LossHistory()
    
    X_train = np.load(train_bottleneck)
    Y_train = np.load(train_labels)
    Y_train = np_utils.to_categorical(Y_train, num_classes=2)
    
    X_test = np.load(test_bottleneck)
    Y_test = np.load(test_labels)
    Y_test = np_utils.to_categorical(Y_test, num_classes=2)

    model = Sequential()
    model.add(Flatten(input_shape=X_train.shape[1:]))
    model.add( Dropout(0.7))
    model.add( Dense(256, activation='relu' ) )
    model.add( Dropout(0.5))
    
    # Softmax for probabilities for each class at the output layer
    model.add( Dense(2, activation='softmax'))
    
    model.compile(optimizer='rmsprop',  # adadelta
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              epochs=epoch,
              batch_size=batch,
              callbacks=[history],
              validation_data=(X_test, Y_test),
              verbose=2)
    
    print ("Training duration : {0}".format(time.time() - start_time))
    score = model.evaluate(X_test, Y_test, batch_size=16, verbose=2)

    print ("Network's test score [loss, accuracy]: {0}".format(score))
    print ('CNN Error: {:.2f}%'.format(100 - score[1] * 100))
    
    #bc.save_model(model_save, model, "jn_VGG16_Detection_top_weights_threshold.h5")
    
    return model, history.losses, history.acc, score

np.random.seed(SEED)
(trans_model, loss_cnn, acc_cnn, test_score_cnn) = train_model(train_feats=train_bottleneck,
                                                                   train_lab=train_labels, 
                                                                   test_feats=test_bottleneck, 
                                                                   test_lab=test_labels,
                                                                   model_path=modelPath, 
                                                                   model_save=top_model_weights_path,
                                                                   epoch=75)

plt.figure(figsize=(10,10))
plot_losses(loss_cnn, acc_cnn)



