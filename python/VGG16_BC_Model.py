import os
import sys
import numpy as np
import keras.callbacks as cb
import keras.utils.np_utils as np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, GaussianNoise
from keras.layers.core import Activation
from keras.constraints import maxnorm
from keras import applications # For easy loading the VGG_16 Model
from skimage import color
import sklearn.metrics as skm
import cv2
# Image loading and other helper functions
import dwdii_bc_model_helper as bc
from matplotlib import pyplot as plt

# Function for rotating the image files.
def Image_Rotate(img, angle):
    """
    Rotates a given image the requested angle. Returns the rotated image.
    """
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
    return(cv2.warpAffine(img,M,(cols,rows)))

# Function for augmenting the images
def Image_Augment(X, Y, vflip=False, hflip=False, major_rotate=False, minor_rotate=False):
    """
    :param  X np.array of images
            Y np.array of labels
            vflip, hflip, major_rotate, minor_rotate set to True to perform the augmentations
    :return The set of augmented iages and their corresponding labels
    
    """
    if len(X) != len(Y):
        print('Data and Label arrays not of the same length.')
    
    n = vflip + hflip + 2*major_rotate + 6*minor_rotate
    augmented = np.zeros([len(X) + n*len(X), X.shape[1], X.shape[2]])
    label = np.zeros([len(Y) + n*len(Y), 1])
    count = 0
    for i in range(0, len(X)):
        augmented[count] = X[i]
        label[count] = Y[i]
        count += 1
        if vflip:
            aug = cv2.flip(X[i], 0)
            augmented[count] = aug
            label[count] = Y[i]
            count += 1
        if hflip:
            aug = cv2.flip(X[i], 1)
            augmented[count] = aug
            label[count] = Y[i]
            count +=1 
        if major_rotate:
            angles = [90, 270]
            for angle in angles:
                aug = Image_Rotate(X[i], angle)
                augmented[count] = aug
                label[count] = Y[i]
                count += 1
        if minor_rotate:
            angles = [-45,-30,-15,15,30,45]
            for angle in angles:
                aug = Image_Rotate(X[i], angle)
                augmented[count] = aug
                label[count] = Y[i]
                count += 1
                
    return(augmented, label)

def VGG_Prep(img_data):
    """
    :param img_data: training or test images of shape [#images, height, width]
    :return: the array transformed to the correct shape for the VGG network
                shape = [#images, height, width, 3] transforms to rgb and reshapes
    """
    images = np.zeros([len(img_data), img_data.shape[1], img_data.shape[2], 3])
    for i in range(0, len(img_data)):
        im = 255 - (img_data[i] * 255)  # Orginal imagnet images were not rescaled
        im = color.gray2rgb(im)
        images[i] = im
    return(images)

def vgg16_bottleneck(trainPath, testPath, imagePath, modelPath, size, balance = True, verbose = True, 
                     verboseFreq = 50, valPath = 'None', transform = False, binary = False):
    
    categories = bc.bcNormVsAbnormNumerics()
    
    # Loading data
    metaTr, metaTr2, mCountsTr = bc.load_training_metadata(trainPath, balance, verbose)
    lenTrain = len(metaTr)
    X_train, Y_train = bc.load_data(trainPath, imagePath, maxData = lenTrain,
                                    categories=categories,
                                    verboseFreq = verboseFreq, 
                                    imgResize=size, 
                                    normalVsAbnormal=binary)
    
    metaTest, meataT2, mCountsT = bc.load_training_metadata(testPath, balance, verbose)
    lenTest = len(metaTest)
    X_test, Y_test = bc.load_data(testPath, imagePath, maxData = lenTrain, 
                                  categories=categories,
                                  verboseFreq = verboseFreq, 
                                  imgResize=size, 
                                  normalVsAbnormal=binary)
    
    if transform:
        print('Transforming the Training Data')
        X_train, Y_train = Image_Augment(X=X_train, Y=Y_train, hflip=True, vflip=True, minor_rotate=False, major_rotate=False)
    
    print('Preparing the Training Data for the VGG_16 Model.')
    X_train = VGG_Prep(X_train)
    print('Preparing the Test Data for the VGG_16 Model')
    X_test = VGG_Prep(X_test)
        
    print('Loading the VGG_16 Model')
    model = applications.VGG16(include_top=False, weights='imagenet')
        
    # Generating the bottleneck features for the training data
    print('Evaluating the VGG_16 Model on the Training Data')
    bottleneck_features_train = model.predict(X_train)
    
    # Saving the bottleneck features for the training data
    featuresTrain = os.path.join(modelPath, 'bottleneck_features_train.npy')
    labelsTrain = os.path.join(modelPath, 'labels_train.npy')
    print('Saving the Training Data Bottleneck Features.')
    np.save(open(featuresTrain, 'wb'), bottleneck_features_train)
    np.save(open(labelsTrain, 'wb'), Y_train)

    # Generating the bottleneck features for the test data
    print('Evaluating the VGG_16 Model on the Test Data')
    bottleneck_features_test = model.predict(X_test)
    
    # Saving the bottleneck features for the test data
    featuresTest = os.path.join(modelPath, 'bottleneck_features_test.npy')
    labelsTest = os.path.join(modelPath, 'labels_test.npy')
    print('Saving the Test Data Bottleneck Feaures.')
    np.save(open(featuresTest, 'wb'), bottleneck_features_test)
    np.save(open(labelsTest, 'wb'), Y_test)
    
    if valPath != 'None':
        metaVal, metaV2, mCountsV = bc.load_training_metadata(valPath, verbose = verbose, balanceViaRemoval = False)
        lenVal = len(metaVal)
        X_val, Y_val = bc.load_data(valPath, imagePath, maxData = lenVal, verboseFreq = verboseFreq, imgResize=size)
        X_val = VGG_Prep(X_val)
        
        # Generating the bottleneck features for the test data
        print('Evaluating the VGG_16 Model on the Validataion Data')
        bottleneck_features_val = model.predict(X_val)
    
        # Saving the bottleneck features for the test data
        featuresVal = os.path.join(modelPath, 'bottleneck_features_validation.npy')
        labelsVal = os.path.join(modelPath, 'labels_validation.npy')
        print('Saving the Validation Data Bottleneck Features.')
        np.save(open(featuresVal, 'wb'), bottleneck_features_val)
        np.save(open(labelsVal, 'wb'), Y_val)

# global variables for loading the data
imagePath = '../images/threshold/DDSM/'
trainDataPath = '../images/ddsm/ddsm_train.csv'
testDataPath = '../images/ddsm/ddsm_test.csv'
valDataPath = '../images/ddsm/ddsm_val.csv'
imgResize = (224, 224) # can go up to (224, 224)
modelPath = '../model/'

vgg16_bottleneck(trainDataPath, testDataPath, imagePath, modelPath, imgResize, 
                 balance = True, verbose = True, verboseFreq = 50, valPath = valDataPath, 
                 transform = False, binary = True)

class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)

def train_top_model(train_feats, train_lab, test_feats, test_lab, model_path, model_save, epoch = 50, batch = 64):
    train_bottleneck = os.path.join(model_path, train_feats)
    train_labels = os.path.join(model_path, train_lab)
    test_bottleneck = os.path.join(model_path, test_feats)
    test_labels = os.path.join(model_path, test_lab)
    
    history = LossHistory()
    
    X_train = np.load(train_bottleneck)
    Y_train = np.load(train_labels)
    #Y_train = np_utils.to_categorical(Y_train, nb_classes=3)
    Y_train = np_utils.to_categorical(Y_train, nb_classes=2)
    
    X_test = np.load(test_bottleneck)
    Y_test = np.load(test_labels)
    #Y_test = np_utils.to_categorical(Y_test, nb_classes=3)
    Y_test = np_utils.to_categorical(Y_test, nb_classes=2)
    print(X_train.shape)
    
    noise = 0.01
    
    model = Sequential()
    model.add( GaussianNoise(noise, input_shape=X_train.shape[1:]))
    model.add(Flatten(input_shape=X_train.shape[1:]))
    model.add(Dropout(0.7))
    model.add( Dense(256, activation = 'relu') )
    model.add(Dropout(0.5))
    #model.add(Dense(3))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    #loss = 'categorical_crossentropy'
    model.compile(optimizer='adadelta',
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              nb_epoch=epoch,
              batch_size=batch,
              callbacks=[history],
              validation_data=(X_test, Y_test),
              verbose=2)
    
    score = model.evaluate(X_test, Y_test, batch_size=16, verbose=0)

    print "Network's test score [loss, accuracy]: {0}".format(score)
    
    model.save_weights(os.path.join(model_path, model_save))

def cf_Matrix(data, label, weights, path, save):
    data = os.path.join(path, data)
    label = os.path.join(path, label)
    categories = bc.bcNormVsAbnormNumerics()
    
    X = np.load(data)
    Y = np.load(label)
    #Y = np_utils.to_categorical(Y, nb_classes=3)
    
    # Loading and preping the model
    model = Sequential()
    model.add(Flatten(input_shape=X.shape[1:]))
    model.add(Dropout(0.7))
    
    model.add(Dense(256))
    model.add(Activation('relu'), constraint= maxnorm(3.))
    model.add(Dropout(0.5))
    
    #model.add(Dense(3))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    
    model.load_weights(os.path.join('../model/', weights))
    
    # try Adadelta and Adam
    model.compile(optimizer='adadelta',
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    predictOutput = model.predict(X, batch_size=64, verbose=2)
    #numBC = bc.numericBC()
    numBC = bc.reverseDict(categories)
    
    predClasses = []
    for i in range(len(predictOutput)):
        arPred = np.array(predictOutput[i])
        predictionProb = arPred.max()
        predictionNdx = arPred.argmax()
        predClassName = numBC[predictionNdx]
        predClasses.append(predictionNdx)
        
    # Use sklearn's helper method to generate the confusion matrix
    cnf_matrix = skm.confusion_matrix(Y, predClasses)
    
    # Ploting the confusion matrix
    class_names = numBC.values()
    np.set_printoptions(precision=2)
    
    fileCfMatrix = '../figures/confusion_matrix-' + save + '.png'
    plt.figure()
    bc.plot_confusion_matrix(cnf_matrix, classes=class_names,
                             title='Confusion matrix, \n' + save)
    plt.savefig(fileCfMatrix)
    plt.show()

# Locations for the bottleneck and labels files that we need
modelPath = '../model/'
train_bottleneck = 'bottleneck_features_train.npy'
train_labels = 'labels_train.npy'
test_bottleneck = 'bottleneck_features_test.npy'
test_labels = 'labels_test.npy'
validation_bottleneck = 'bottleneck_features_valdation.npy'
validation_label = 'labels_validation.npy'
top_model_weights_path = 'top_weights02.h5'

train_top_model(train_feats=train_bottleneck, train_lab=train_labels, test_feats=test_bottleneck, test_lab=test_labels,
                model_path=modelPath, model_save=top_model_weights_path)

feats_loc = '150_test_val/bottleneck_features_test.npy'
feats_labs = '150_test_val/labels_test.npy'
weight = 'balanced150run2/top_weights02.h5'
saveFile = 'balanced150'

cf_Matrix(data=feats_loc, label=feats_labs, weights=weight, path=modelPath, save=saveFile)

# Locations for the bottleneck and labels files that we need
modelPath = '../model/'
train_bottleneck = 'bottleneck_features_150fulltrans_train.npy'
train_labels = 'labels_150fulltrans_train.npy'
test_bottleneck = 'bottleneck_features_test.npy'
test_labels = 'labels_test.npy'
validation_bottleneck = 'bottleneck_features_valdation.npy'
validation_label = 'labels_validation.npy'
top_model_weights_path = 'top_weights_150fulltrans.h5'

train_top_model(train_feats=train_bottleneck, train_lab=train_labels, test_feats=test_bottleneck, test_lab=test_labels,
                model_path=modelPath, model_save=top_model_weights_path)

feats_loc = '150_test_val/bottleneck_features_test.npy'
feats_labs = '150_test_val/labels_test.npy'
weight = 'balanced150FullTrans/top_weights_150fulltrans.h5'
saveFile = 'balanced150FullTrans'

cf_Matrix(data=feats_loc, label=feats_labs, weights=weight, path=modelPath, save=saveFile)

# Locations for the bottleneck and labels files that we need
modelPath = '../model/'
train_bottleneck = 'bottleneck_features_train_224.npy'
train_labels = 'labels_train_224.npy'
test_bottleneck = 'bottleneck_features_test.npy'
test_labels = 'labels_test.npy'
validation_bottleneck = 'bottleneck_features_valdation.npy'
validation_label = 'labels_validation.npy'
top_model_weights_path = 'top_weights_224.h5'

train_top_model(train_feats=train_bottleneck, train_lab=train_labels, test_feats=test_bottleneck, test_lab=test_labels,
                model_path=modelPath, model_save=top_model_weights_path)

feats_loc = '224_test_val/bottleneck_features_test.npy'
feats_labs = '224_test_val/labels_test.npy'
weight = 'balanced224/top_weights_224.h5'
saveFile = 'balanced224'

cf_Matrix(data=feats_loc, label=feats_labs, weights=weight, path=modelPath, save=saveFile)

# Locations for the bottleneck and labels files that we need
modelPath = '../model/'
train_bottleneck = 'Balanced224flips/bottleneck_features_train_224flip.npy'
train_labels = 'Balanced224flips/labels_train_224flip.npy'
test_bottleneck = '224_test_val/bottleneck_features_test.npy'
test_labels = '224_test_val/labels_test.npy'
validation_bottleneck = 'bottleneck_features_valdation.npy'
validation_label = 'labels_validation.npy'
top_model_weights_path = 'Balanced224flips/top_weights_224flip.h5'

train_top_model(train_feats=train_bottleneck, train_lab=train_labels, test_feats=test_bottleneck, test_lab=test_labels,
                model_path=modelPath, model_save=top_model_weights_path)

feats_loc = '224_test_val/bottleneck_features_test.npy'
feats_labs = '224_test_val/labels_test.npy'
weight = 'balanced224flips/top_weights_224flip.h5'
saveFile = 'balanced224flip'

cf_Matrix(data=feats_loc, label=feats_labs, weights=weight, path=modelPath, save=saveFile)

# Locations for the bottleneck and labels files that we need
modelPath = '../model/'
train_bottleneck = 'bottleneck_features_train_224th.npy'
train_labels = 'labels_train_224th.npy'
test_bottleneck = 'bottleneck_features_test.npy'
test_labels = 'labels_test.npy'
validation_bottleneck = 'bottleneck_features_valdation.npy'
validation_label = 'labels_validation.npy'
top_model_weights_path = 'top_weights_224th.h5'

train_top_model(train_feats=train_bottleneck, train_lab=train_labels, test_feats=test_bottleneck, test_lab=test_labels,
                model_path=modelPath, model_save=top_model_weights_path)

feats_loc = '224_threshold/bottleneck_features_test.npy'
feats_labs = '224_threshold/labels_test.npy'
weight = 'balanced224Threshold/top_weights_224th.h5'
saveFile = 'balanced224Threshold'

cf_Matrix(data=feats_loc, label=feats_labs, weights=weight, path=modelPath, save=saveFile)

# Locations for the bottleneck and labels files that we need
modelPath = '../model/'
train_bottleneck = 'Balanced224Binary/bottleneck_features_train_224twoclass.npy'
train_labels = 'Balanced224Binary/labels_train_224twoclass.npy'
test_bottleneck = '224_binary/bottleneck_features_test.npy'
test_labels = '224_binary/labels_test.npy'
validation_bottleneck = 'bottleneck_features_valdation.npy'
validation_label = 'labels_validation.npy'
top_model_weights_path = 'Balanced224Binary/top_weights_224twoclass.h5'

train_top_model(train_feats=train_bottleneck, train_lab=train_labels, test_feats=test_bottleneck, test_lab=test_labels,
                model_path=modelPath, model_save=top_model_weights_path, epoch = 100)

feats_loc = '224_binary/bottleneck_features_test.npy'
feats_labs = '224_binary/labels_test.npy'
weight = 'balanced224Binary/top_weights_224twoclass.h5'
saveFile = 'balanced224Twoclass'

cf_Matrix(data=feats_loc, label=feats_labs, weights=weight, path=modelPath, save=saveFile)

# Locations for the bottleneck and labels files that we need
modelPath = '../model/'
train_bottleneck = 'bottleneck_features_train_224th_twoclass.npy'
train_labels = 'labels_train_224th_twoclass.npy'
test_bottleneck = 'bottleneck_features_test.npy'
test_labels = 'labels_test.npy'
validation_bottleneck = 'bottleneck_features_valdation.npy'
validation_label = 'labels_validation.npy'
top_model_weights_path = 'top_weights_224th_twoclass.h5'

train_top_model(train_feats=train_bottleneck, train_lab=train_labels, test_feats=test_bottleneck, test_lab=test_labels,
                model_path=modelPath, model_save=top_model_weights_path)

feats_loc = '224_binary/bottleneck_features_test.npy'
feats_labs = '224_binary/labels_test.npy'
weight = 'balanced224Th_Binary/top_weights_224th_twoclass.h5'
saveFile = 'balanced224Th_Twoclass'

cf_Matrix(data=feats_loc, label=feats_labs, weights=weight, path=modelPath, save=saveFile)

