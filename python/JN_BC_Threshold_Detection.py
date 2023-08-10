import os
import sys
import time
import numpy as np

from tqdm import tqdm

import sklearn.metrics as skm
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from skimage import color

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
get_ipython().magic('matplotlib inline')

plt.rcParams['figure.figsize'] = (10,10)
np.set_printoptions(precision=2)

sys.path.insert(0, '../helper_modules/')
import jn_bc_helper as bc

get_ipython().run_cell_magic('python', '', "import os\nos.system('python -V')\nos.system('python ../helper_modules/Package_Versions.py')")

SEED = 7
np.random.seed(SEED)

CURR_DIR  = os.getcwd()
DATA_DIR  = '/Users/jnarhan/Dropbox/Breast_Cancer_Data/Data_Thresholded/ALL_IMGS/'
AUG_DIR   = '/Users/jnarhan/Dropbox/Breast_Cancer_Data/Data_Thresholded/AUG_DETECT_IMGS/'
meta_file = '../../Meta_Data_Files/meta_data_all.csv'
PATHO_INX = 5    # Column number of detected label in meta_file
FILE_INX  = 1    # Column number of File name in meta_file

meta_data, _ = tqdm( bc.load_meta(meta_file, patho_idx=PATHO_INX, file_idx=FILE_INX,
                                  balanceByRemoval=False, verbose=False) )

# Minor addition to reserve records in meta data for which we actually have images:
meta_data = bc.clean_meta(meta_data, DATA_DIR)

bc.pprint('Loading data')
cats = bc.bcLabels(['normal', 'abnormal'])

# For smaller images supply tuple argument for a parameter 'imgResize':
# X_data, Y_data = bc.load_data(meta_data, DATA_DIR, cats, imgResize=(150,150)) 
X_data, Y_data = tqdm( bc.load_data(meta_data, DATA_DIR, cats) )

cls_cnts = bc.get_clsCnts(Y_data, cats)
bc.pprint('Before Balancing')
for k in cls_cnts:
    print '{0:10}: {1}'.format(k, cls_cnts[k])

datagen = ImageDataGenerator(rotation_range=5, width_shift_range=.01, height_shift_range=0.01,
                             data_format='channels_first')

X_data, Y_data = bc.balanceViaSmote(cls_cnts, meta_data, DATA_DIR, AUG_DIR, cats, 
                                    datagen, X_data, Y_data, seed=SEED, verbose=True)

X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data,
                                                    test_size=0.25,
                                                    random_state=SEED,
                                                    stratify=zip(*Y_data)[0])

print 'Size of X_train: {:>5}'.format(len(X_train))
print 'Size of X_test: {:>5}'.format(len(X_test))
print 'Size of Y_train: {:>5}'.format(len(Y_train))
print 'Size of Y_test: {:>5}'.format(len(Y_test))

print X_train.shape
print X_test.shape
print Y_train.shape
print Y_test.shape

data = [X_train, X_test, Y_train, Y_test]

X_train_svm = X_train.reshape( (X_train.shape[0], -1)) 
X_test_svm  = X_test.reshape( (X_test.shape[0], -1))

SVM_model = SVC(gamma=0.001)
SVM_model.fit( X_train_svm, Y_train)

predictOutput = SVM_model.predict(X_test_svm)
svm_acc = metrics.accuracy_score(y_true=Y_test, y_pred=predictOutput)

print 'SVM Accuracy: {: >7.2f}%'.format(svm_acc * 100)
print 'SVM Error: {: >10.2f}%'.format(100 - svm_acc * 100)

svm_matrix = skm.confusion_matrix(y_true=Y_test, y_pred=predictOutput)
numBC = bc.reverseDict(cats)
class_names = numBC.values()

plt.figure(figsize=(8,6))
bc.plot_confusion_matrix(svm_matrix, classes=class_names, normalize=True, 
                         title='SVM Normalized Confusion Matrix Using Thresholded \n')
plt.tight_layout()
plt.savefig('../../figures/jn_SVM_Detect_Threshold_CM_20170609.png', dpi=100)

plt.figure(figsize=(8,6))
bc.plot_confusion_matrix(svm_matrix, classes=class_names, normalize=False, 
                         title='SVM Raw Confusion Matrix Using Thresholded \n')
plt.tight_layout()

bc.cat_stats(svm_matrix)

def VGG_Prep(img_data):
    """
    :param img_data: training or test images of shape [#images, height, width]
    :return: the array transformed to the correct shape for the VGG network
                shape = [#images, height, width, 3] transforms to rgb and reshapes
    """
    images = np.zeros([len(img_data), img_data.shape[1], img_data.shape[2], 3])
    for i in range(0, len(img_data)):
        im = (img_data[i] * 255)        # Original imagenet images were not rescaled
        im = color.gray2rgb(im)
        images[i] = im
    return(images)

def vgg16_bottleneck(data, modelPath, fn_train_feats, fn_train_lbls, fn_test_feats, fn_test_lbls):
    # Loading data
    X_train, X_test, Y_train, Y_test = data
    
    print('Preparing the Training Data for the VGG_16 Model.')
    X_train = VGG_Prep(X_train)
    print('Preparing the Test Data for the VGG_16 Model')
    X_test = VGG_Prep(X_test)
        
    print('Loading the VGG_16 Model')
    # "model" excludes top layer of VGG16:
    model = applications.VGG16(include_top=False, weights='imagenet') 
        
    # Generating the bottleneck features for the training data
    print('Evaluating the VGG_16 Model on the Training Data')
    bottleneck_features_train = model.predict(X_train)
    
    # Saving the bottleneck features for the training data
    featuresTrain = os.path.join(modelPath, fn_train_feats)
    labelsTrain = os.path.join(modelPath, fn_train_lbls)
    print('Saving the Training Data Bottleneck Features.')
    np.save(open(featuresTrain, 'wb'), bottleneck_features_train)
    np.save(open(labelsTrain, 'wb'), Y_train)

    # Generating the bottleneck features for the test data
    print('Evaluating the VGG_16 Model on the Test Data')
    bottleneck_features_test = model.predict(X_test)
    
    # Saving the bottleneck features for the test data
    featuresTest = os.path.join(modelPath, fn_test_feats)
    labelsTest = os.path.join(modelPath, fn_test_lbls)
    print('Saving the Test Data Bottleneck Feaures.')
    np.save(open(featuresTest, 'wb'), bottleneck_features_test)
    np.save(open(labelsTest, 'wb'), Y_test)

# Locations for the bottleneck and labels files that we need
train_bottleneck = '2Class_VGG16_bottleneck_features_train_threshold.npy'
train_labels     = '2Class_VGG16_labels_train_threshold.npy'
test_bottleneck  = '2Class_VGG16_bottleneck_features_test_threshold.npy'
test_labels      = '2Class_VGG16_labels_test_threshold.npy'
modelPath = os.getcwd()

top_model_weights_path = './weights/'

np.random.seed(SEED)
vgg16_bottleneck(data, modelPath, train_bottleneck, train_labels, test_bottleneck, test_labels)

def train_top_model(train_feats, train_lab, test_feats, test_lab, model_path, model_save, epoch = 50, batch = 64):
    start_time = time.time()
    
    train_bottleneck = os.path.join(model_path, train_feats)
    train_labels = os.path.join(model_path, train_lab)
    test_bottleneck = os.path.join(model_path, test_feats)
    test_labels = os.path.join(model_path, test_lab)
    
    history = bc.LossHistory()
    
    X_train = np.load(train_bottleneck)
    Y_train = np.load(train_labels)
    Y_train = np_utils.to_categorical(Y_train, num_classes=2)
    
    X_test = np.load(test_bottleneck)
    Y_test = np.load(test_labels)
    Y_test = np_utils.to_categorical(Y_test, num_classes=2)

    model = Sequential()
    model.add(Flatten(input_shape=X_train.shape[1:]))
    model.add( Dropout(0.7))
    
    model.add( Dense(256, activation='relu', kernel_constraint= maxnorm(3.)) )
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
    
    print "Training duration : {0}".format(time.time() - start_time)
    score = model.evaluate(X_test, Y_test, batch_size=16, verbose=2)

    print "Network's test score [loss, accuracy]: {0}".format(score)
    print 'CNN Error: {:.2f}%'.format(100 - score[1] * 100)
    
    bc.save_model(model_save, model, "jn_VGG16_Detection_top_weights_threshold.h5")
    
    return model, history.losses, history.acc, score

np.random.seed(SEED)
(trans_model, loss_cnn, acc_cnn, test_score_cnn) = train_top_model(train_feats=train_bottleneck,
                                                                   train_lab=train_labels, 
                                                                   test_feats=test_bottleneck, 
                                                                   test_lab=test_labels,
                                                                   model_path=modelPath, 
                                                                   model_save=top_model_weights_path,
                                                                   epoch=100)
plt.figure(figsize=(10,10))
bc.plot_losses(loss_cnn, acc_cnn)
plt.savefig('../../figures/epoch_figures/jn_Transfer_Detection_Learning_Threshold_20170609.png', dpi=100)

print 'Transfer Learning CNN Accuracy: {: >7.2f}%'.format(test_score_cnn[1] * 100)
print 'Transfer Learning CNN Error: {: >10.2f}%'.format(100 - test_score_cnn[1] * 100)

predictOutput = bc.predict(trans_model, np.load(test_bottleneck))
trans_matrix = skm.confusion_matrix(y_true=Y_test, y_pred=predictOutput)

plt.figure(figsize=(8,6))
bc.plot_confusion_matrix(trans_matrix, classes=class_names, normalize=True,
                         title='Transfer CNN Normalized Confusion Matrix Using Thresholded \n')
plt.tight_layout()
plt.savefig('../../figures/jn_Transfer_Detection_CM_Threshold_20170609.png', dpi=100)

plt.figure(figsize=(8,6))
bc.plot_confusion_matrix(trans_matrix, classes=class_names, normalize=False,
                         title='Transfer CNN Raw Confusion Matrix Using Thresholded \n')
plt.tight_layout()

bc.cat_stats(trans_matrix)

data = [X_train, X_test, Y_train, Y_test]
X_train, X_test, Y_train, Y_test = bc.prep_data(data, cats)
data = [X_train, X_test, Y_train, Y_test]

print X_train.shape
print X_test.shape
print Y_train.shape
print Y_test.shape

def diff_model_v7_reg(numClasses, input_shape=(3, 150,150), add_noise=False, noise=0.01, verbose=False):
    model = Sequential()
    if (add_noise):
        model.add( GaussianNoise(noise, input_shape=input_shape))
        model.add( Convolution2D(filters=16, 
                                 kernel_size=(5,5), 
                                 data_format='channels_first',
                                 padding='same',
                                 activation='relu'))
    else:
        model.add( Convolution2D(filters=16, 
                                 kernel_size=(5,5), 
                                 data_format='channels_first',
                                 padding='same',
                                 activation='relu',
                                 input_shape=input_shape))
    model.add( Dropout(0.7))
    
    model.add( Convolution2D(filters=32, kernel_size=(3,3), 
                             data_format='channels_first', padding='same', activation='relu'))
    model.add( MaxPooling2D(pool_size= (2,2), data_format='channels_first'))
    model.add( Dropout(0.4))
    model.add( Convolution2D(filters=32, kernel_size=(3,3), 
                             data_format='channels_first', activation='relu'))
    
    model.add( Convolution2D(filters=64, kernel_size=(3,3), 
                             data_format='channels_first', padding='same', activation='relu',
                             kernel_regularizer=regularizers.l2(0.01)))
    model.add( MaxPooling2D(pool_size= (2,2), data_format='channels_first'))
    model.add( Convolution2D(filters=64, kernel_size=(3,3), 
                             data_format='channels_first', activation='relu',
                             kernel_regularizer=regularizers.l2(0.01)))
    model.add( Dropout(0.4))
    
    model.add( Convolution2D(filters=128, kernel_size=(3,3), 
                             data_format='channels_first', padding='same', activation='relu',
                             kernel_regularizer=regularizers.l2(0.01)))
    model.add( MaxPooling2D(pool_size= (2,2), data_format='channels_first'))
    
    model.add( Convolution2D(filters=128, kernel_size=(3,3), 
                             data_format='channels_first', activation='relu',
                             kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.4))
    
    model.add( Flatten())
    
    model.add( Dense(128, activation='relu', kernel_constraint= maxnorm(3.)) )
    model.add( Dropout(0.4))
    
    model.add( Dense(64, activation='relu', kernel_constraint= maxnorm(3.)) )
    model.add( Dropout(0.4))
    
    # Softmax for probabilities for each class at the output layer
    model.add( Dense(numClasses, activation='softmax'))
    
    if verbose:
        print( model.summary() )
    
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

diff_model7_noise_reg = diff_model_v7_reg(len(cats),
                                          input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]),
                                          add_noise=True, verbose=True)

np.random.seed(SEED)

(cnn_model, loss_cnn, acc_cnn, test_score_cnn) = bc.run_network(model=diff_model7_noise_reg, 
                                                                data=data, 
                                                                epochs=50, batch=64)
plt.figure(figsize=(10,10))
bc.plot_losses(loss_cnn, acc_cnn)
plt.savefig('../../figures/epoch_figures/jn_Core_CNN_Detection_Learning_Threshold_20170609.png', dpi=100)

bc.save_model(dir_path='./weights/', model=cnn_model, name='jn_Core_CNN_Detection_Threshold_20170609')

print 'Core CNN Accuracy: {: >7.2f}%'.format(test_score_cnn[1] * 100)
print 'Core CNN Error: {: >10.2f}%'.format(100 - test_score_cnn[1] * 100)

predictOutput = bc.predict(cnn_model, X_test)

cnn_matrix = skm.confusion_matrix(y_true=[val.argmax() for val in Y_test], y_pred=predictOutput)

plt.figure(figsize=(8,6))
bc.plot_confusion_matrix(cnn_matrix, classes=class_names, normalize=True,
                         title='CNN Normalized Confusion Matrix Using Thresholded \n')
plt.tight_layout()
plt.savefig('../../figures/jn_Core_CNN_Detection_CM_Threshold_20170609.png', dpi=100)

plt.figure(figsize=(8,6))
bc.plot_confusion_matrix(cnn_matrix, classes=class_names, normalize=False,
                         title='CNN Raw Confusion Matrix Using Thresholded \n')
plt.tight_layout()

bc.cat_stats(cnn_matrix)



