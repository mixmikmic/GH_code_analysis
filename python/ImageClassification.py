import numpy as np
import cv2
import os
import pandas as pd
from matplotlib import pyplot as plt
import skimage.feature
import imutils

# Loop through each image
def imgProcess( filePath, dottedFilePath, walkSize ):
    imgs = []
    labels = []
    file_names = os.listdir(filePath)
    print(file_names)
    for filename in file_names:
        image_1 = cv2.imread(dottedFilePath + filename)
        image_2 = cv2.imread(filePath + filename)
        height, width = image_1.shape[:2]
        cut = np.copy(image_2)

        # absolute difference between the orignal and the dotted
        image_3 = cv2.absdiff(image_1,image_2)
        # mask out blackened regions from the dotted images
        mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        mask_1[mask_1 < 20] = 0
        mask_1[mask_1 > 0] = 255

        mask_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
        mask_2[mask_2 < 20] = 0
        mask_2[mask_2 > 0] = 255

        image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_1)
        image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_2) 

        # convert to grayscale to be accepted by skimage.feature.blob_log
        image_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2GRAY)

        # Detect the dotted defects in the dotted image
        blobs = skimage.feature.blob_log(image_3, min_sigma=6, max_sigma=8, num_sigma=2, threshold=0.04, overlap = 0.5)

        dents = []
        image_circles = image_1

        # For each file, identify the coordinates of all defects based on blobs
        for blob in blobs:
            # get the coordinates for each blob
            y, x, s = blob

            # get the color of the pixel from Train Dotted in the center of the blob
            g,b,r = image_1[int(y)][int(x)][:]

            # detect whether this is the defects center
            if r > 240 and g < 20 and b < 20: # RED
                dents.append((int(x),int(y)))
                cv2.circle(image_circles, (int(x),int(y)), 20, (0,0,255), 10)    
  
            cv2.rectangle(cut, (int(x)-32,int(y)-32),(int(x)+32,int(y)+32), 0,-1)  

        # For each blob, find the positive case
        for coordinates in dents:        
            thumb = image_2[coordinates[1]-32:coordinates[1]+32,coordinates[0]-32:coordinates[0]+32,:]
            if np.shape(thumb) == (64, 64, 3):
                imgs.append(thumb)
                labels.append('1')
        # Expand the samples by finding more negative samples           
        for i in range(0,np.shape(cut)[0],walkSize):
            for j in range(0,np.shape(cut)[1],walkSize):                
                thumb = cut[i:i+64,j:j+64,:]
                if np.amin(cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)) != 0:
                    if np.shape(thumb) == (64,64,3):
                        imgs.append(thumb)
                        labels.append('0')
    return imgs, labels

filePath = "images/Train/"
dottedFilePath = "images/Train_dotted/"
walkSize = 300
trainx, trainy = imgProcess(filePath, dottedFilePath, walkSize)

print(len(trainx))
print(len(trainy))
print(trainy)

originx = np.copy(trainx)
originy = np.copy(trainy)
trainx = []
trainy = []   
    
for i in range(0,len(originx)):
    sample_image = originx[i]
    for rotated_angle in np.arange(0, 360, 15):
        rotated = imutils.rotate(sample_image, rotated_angle)
        trainx.append(rotated)
        trainy.append(originy[i])
        for flipped_angle in np.arange(0, 2):
            flipped_image = cv2.flip(rotated, flipped_angle)
            trainx.append(flipped_image)
            trainy.append(originy[i])

print((trainx[2].shape))
print(len(trainy))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, Cropping2D, Input
from keras.utils import np_utils
from keras import optimizers
from collections import Counter
print(keras.__version__)

trainx = np.array(trainx)
trainy = np.array(trainy)

encoder = LabelBinarizer()
encoder.fit(trainy)
trainy = keras.utils.to_categorical(trainy, num_classes=2)

print(len(trainx), len(trainy))
print (trainx.shape)
print(trainy.shape)

filePath = "images/Test/"
dottedFilePath = "images/Test_dotted/"
walkSize = 300
testx, testy = imgProcess(filePath, dottedFilePath, walkSize)
testx = np.array(testx)
testy = np.array(testy)

print(len(testx), len(testy))
print (testx.shape)
print(testy.shape)

print(testy)

def createModel():    
    model = Sequential()

    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(64,64,3)))


    model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(256, (5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    opt = optimizers.Adam(decay = 0.1)
    model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
    return model

mymodel = createModel()
seed = 123
numEpochs = 30
batchSize = 16
X_train, X_test, y_train, y_test = train_test_split(trainx, trainy, test_size=0.25, random_state=seed)
myhistory = mymodel.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=numEpochs, batch_size = batchSize, verbose=0)

def plotPerf(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

plotPerf(myhistory)

y_predicted = mymodel.predict(testx, verbose = 0)
y_pred = encoder.inverse_transform(y_predicted)
print("Predicted: \n", y_pred)
print("Actual: \n",testy)

print(testy[5])
plt.imshow(cv2.cvtColor(testx[5], cv2.COLOR_BGR2RGB))
plt.show()

from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
import pandas as pd
from ggplot import *

label_y = [int(numeric_string) for numeric_string in testy]
pred_y = [int(numeric_string) for numeric_string in y_pred]

# Confusion matrix
confusion = confusion_matrix(label_y, pred_y)
print(confusion)
# Precision 
precision_val = precision_score(label_y, pred_y, pos_label = 1)
# Recall
recall_val = recall_score(label_y, pred_y)
# F1 score
f1_val = f1_score(label_y,pred_y)

print("Precision: ", precision_val)
print("Recall   :", recall_val)
print("F1 score :", f1_val)

# Calculate AUC
fpr, tpr, _ = metrics.roc_curve(label_y, y_predicted[:,1])
auc = metrics.auc(fpr,tpr)
print("AUC      :", auc)

# Plot ROC curve
preds = y_predicted[:,1]
fpr, tpr, _ = metrics.roc_curve(label_y, preds)
df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
ggplot(df, aes(x='fpr', y='tpr')) +    geom_line() +    geom_abline(linetype='dashed') +    ggtitle("ROC Curve w/ AUC=%s" % str(auc))

from keras import applications
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

img_rows, img_cols, img_channel = 64, 64, 3
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_channel))

# Freeze the layers which you don't want to train. Here I am freezing the first 10 layers.
for layer in base_model.layers[:10]:
    layer.trainable = False

def createVGG16Model():
    add_model = Sequential()
    add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    add_model.add(Dense(512, activation='relu'))
    add_model.add(Dense(256, activation='relu'))
    add_model.add(Dense(2, activation='softmax'))

    model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
    opt = optimizers.SGD(lr=1e-4, momentum=0.9)
    model.compile(loss='binary_crossentropy', 
                  optimizer=opt,
                  metrics=['accuracy'])

    model.summary()
    return model

vgg16model = createVGG16Model()
vgghistory = vgg16model.fit(trainx, 
                           trainy, 
                           validation_split=0.25, 
                           epochs=10, 
                           batch_size=batchSize, 
                           verbose=1)

plotPerf(vgghistory)

y_predicted = vgg16model.predict(testx, verbose=0)
y_pred = encoder.inverse_transform(y_predicted)
print("Predicted: \n", y_pred)
print("Actual: \n",testy)

print("Prediction: ", y_pred)
print("Acutal    : ", testy)
label_y = [int(numeric_string) for numeric_string in testy]
pred_y = [int(numeric_string) for numeric_string in y_pred]

# Confusion matrix
confusion = confusion_matrix(label_y, pred_y)
print(confusion)
# Precision 
precision_val = precision_score(label_y, pred_y, pos_label = 1)
# Recall
recall_val = recall_score(label_y, pred_y)
# F1 score
f1_val = f1_score(label_y,pred_y)

print("Precision: ", precision_val)
print("Recall   :", recall_val)
print("F1 score :", f1_val)


fpr, tpr, _ = metrics.roc_curve(label_y, y_predicted[:,1])
auc = metrics.auc(fpr,tpr)
print("AUC      :", auc)

preds = y_predicted[:,1]
fpr, tpr, _ = metrics.roc_curve(label_y, preds)
df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
ggplot(df, aes(x='fpr', y='tpr')) +    geom_line() +    geom_abline(linetype='dashed') +    ggtitle("ROC Curve w/ AUC=%s" % str(auc))

import h5py
model_json_string=mymodel.to_json()
open('my_model_architecture.json','w').write(model_json_string)
mymodel.save_weights('my_model_weights.h5')
print('saved!')



