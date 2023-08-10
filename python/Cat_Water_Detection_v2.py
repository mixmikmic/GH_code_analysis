from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import SGD
from keras.regularizers import l2
import matplotlib.image as mpimg
from scipy.misc import imresize
from scipy.misc import imread
import scipy.misc
import numpy as np
import keras.backend as K
import math
import os
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop
import tensorflow as tf
import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing import image

from ssd import SSD300
from ssd_utils import BBoxUtility

K.clear_session()

img_size=299
train_num = 160
test_num = 36

train_base_dir = './data/v2/train'
test_base_dir = './data/v2/test'

train_not_empty_filenames = os.listdir(train_base_dir + '/0')
train_empty_filenames = os.listdir(train_base_dir + '/1')

test_not_empty_filenames = os.listdir(test_base_dir + '/0')
test_empty_filenames = os.listdir(test_base_dir + '/1')

#print("Train not empty")
# for i, filename in enumerate(train_not_empty_filenames[:2]):
#     print(filename)
#     img = mpimg.imread(train_base_dir + '/0/' + filename)
#     fig, ax = plt.subplots()
#     ax.imshow(img)
#     plt.show()

# print("Train empty")
# for i, filename in enumerate(train_empty_filenames[:2]):
#     print(filename)
#     img = mpimg.imread(train_base_dir + '/1/' + filename)
#     fig, ax = plt.subplots()
#     ax.imshow(img)
#     plt.show()

# print("Test not empty")
# for i, filename in enumerate(test_not_empty_filenames[:2]):
#     print(filename)
#     img = mpimg.imread(test_base_dir + '/0/' + filename)
#     fig, ax = plt.subplots()
#     ax.imshow(img)
#     plt.show()

# print("Test empty")
# for i, filename in enumerate(test_empty_filenames[:2]):
#     print(filename)
#     img = mpimg.imread(test_base_dir + '/1/' + filename)
#     fig, ax = plt.subplots()
#     ax.imshow(img)
#     plt.show()

#Data Augmentation
train_datagen = ImageDataGenerator(
        featurewise_center = False,
        samplewise_center = False,
        featurewise_std_normalization = False,
        samplewise_std_normalization = False,
        rotation_range = 2,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        horizontal_flip = True,
        vertical_flip = False,
        zoom_range = 0.1,
        channel_shift_range = 30,
        fill_mode = 'reflect')

#Test data should not be augmented
test_datagen = ImageDataGenerator()

voc_classes = ['fountain', 'cat']
NUM_CLASSES = len(voc_classes) + 1
input_shape=(300, 300, 3)
model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights('./model/weights.18-0.09.hdf5', by_name=True)
bbox_util = BBoxUtility(NUM_CLASSES)

def ssd_image(img, results, i):
    feeder_list = ['fountain']
    feeder_resolution = np.zeros(1)
    feeder_in_single_image = []

    # Parse the outputs.
    det_label = results[i][:, 0]
    det_conf = results[i][:, 1]
    det_xmin = results[i][:, 2]
    det_ymin = results[i][:, 3]
    det_xmax = results[i][:, 4]
    det_ymax = results[i][:, 5]

    # Get detections with confidence higher than 0.5.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.5]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    for j in range(top_conf.shape[0]):
        xmin = int(round(top_xmin[j] * img.shape[1]))
        ymin = int(round(top_ymin[j] * img.shape[0]))
        xmax = int(round(top_xmax[j] * img.shape[1]))
        ymax = int(round(top_ymax[j] * img.shape[0]))
        score = top_conf[j]
        label = int(top_label_indices[j])
        label_name = voc_classes[label - 1]

        if label_name in feeder_list:
            resolution = (ymax-ymin)*(xmax-xmin)

            #if the detected object is bigger than 400 pixels (e.g. 20 x 20)
            if resolution >= 400:
                feeder_resolution = np.append(feeder_resolution, resolution)

                cropped_img = img[ymin:ymax, xmin:xmax, :]
                feeder_in_single_image.append(cropped_img)

    if len(feeder_in_single_image) > 0:
        max_resolution_index = np.argmax(feeder_resolution)
        #plt.imshow(feeder_in_single_image[max_resolution_index-1] / 255.)
        #plt.show();
        ssd_img = np.uint8(feeder_in_single_image[max_resolution_index-1])
    else:
        ssd_img = img

    return ssd_img
        
#Load images
def load_images(base_dir, num):
    all_imgs = []
    all_classes = []

    #Load not empty images with label 0
    for i in range(num):
        inputs = []
        images = []
    
        img_name = base_dir + '/0/'+ str(i + 1) + '.jpg'
        img = image.load_img(img_name, target_size=(300, 300))
        img = image.img_to_array(img)
        images.append(imread(img_name))
        inputs.append(img.copy())
        inputs = preprocess_input(np.array(inputs))
        preds = model.predict(inputs, batch_size=1, verbose=0)
        results = bbox_util.detection_out(preds)

        for i, img in enumerate(images):
            if type(results[i]) is not list:
                ssd_img = ssd_image(img, results, i)
                
        resize_img = imresize(ssd_img, (img_size, img_size))
        all_imgs.append(resize_img)
        all_classes.append(0)
    
    #Load not empty images with label 1
    for i in range(num):
        inputs = []
        images = []
    
        img_name = base_dir + '/1/'+ str(i + 1) + '.jpg'
        img = image.load_img(img_name, target_size=(300, 300))
        img = image.img_to_array(img)
        images.append(imread(img_name))
        inputs.append(img.copy())
        inputs = preprocess_input(np.array(inputs))
        preds = model.predict(inputs, batch_size=1, verbose=0)
        results = bbox_util.detection_out(preds)

        for i, img in enumerate(images):
            if type(results[i]) is not list:
                ssd_img = ssd_image(img, results, i)
                
        resize_img = imresize(ssd_img, (img_size, img_size))
        all_imgs.append(resize_img)
        all_classes.append(1)
        
    return np.array(all_imgs), np.array(all_classes)

x_train, y_train = load_images(train_base_dir, train_num)
x_test, y_test = load_images(test_base_dir, test_num)

train_generator = train_datagen.flow(x_train, y_train, batch_size = 64, seed = 13)
test_generator = test_datagen.flow(x_test, y_test, batch_size = 64, seed = 13)

#Load Inception v3 model except for the final layer
base_model = InceptionV3(weights = 'imagenet', include_top = False)
#Final layer setting
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(1, kernel_initializer = "glorot_uniform", activation = "sigmoid", kernel_regularizer = l2(.0005))(x)

model = Model(inputs = base_model.input, outputs = predictions)

#Make the model non-trainable since we won't update the weights of the pre-trained model druing training
for layer in base_model.layers:
    layer.trainable = False

#opt = RMSprop(lr = 0.001)
opt = SGD(lr = 0.01, momentum = 0.9)
model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['acc'])

checkpointer = ModelCheckpoint(filepath = './model/model_v2.{epoch:02d}-{val_loss:.2f}.hdf5', verbose = 1, save_best_only = True)
csv_logger = CSVLogger('./model/model_v2.log')

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2,
                  patience = 5, min_lr = 0.001)

history = model.fit_generator(train_generator,
                    steps_per_epoch = train_num * 2,
                    epochs = 5,
                    validation_data = test_generator,
                    validation_steps = test_num * 2,
                    verbose = 0,
                    callbacks = [reduce_lr, csv_logger, checkpointer])

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')
plt.show()

plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')
plt.show()

from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.misc import imresize
from scipy.misc import imread
import numpy as np
import tensorflow as tf
import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing import image

from ssd import SSD300
from ssd_utils import BBoxUtility

model_cnn = load_model(filepath='./model/model_v2.03-0.40.hdf5')

img_size=299
num_test = 32

voc_classes = ['fountain', 'cat']
NUM_CLASSES = len(voc_classes) + 1
input_shape=(300, 300, 3)
model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights('./model/weights.18-0.09.hdf5', by_name=True)
bbox_util = BBoxUtility(NUM_CLASSES)

def ssd_image(img, results, i):
    feeder_list = ['fountain']
    feeder_resolution = np.zeros(1)
    feeder_in_single_image = []

    # Parse the outputs.
    det_label = results[i][:, 0]
    det_conf = results[i][:, 1]
    det_xmin = results[i][:, 2]
    det_ymin = results[i][:, 3]
    det_xmax = results[i][:, 4]
    det_ymax = results[i][:, 5]

    # Get detections with confidence higher than 0.5.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.5]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    for j in range(top_conf.shape[0]):
        xmin = int(round(top_xmin[j] * img.shape[1]))
        ymin = int(round(top_ymin[j] * img.shape[0]))
        xmax = int(round(top_xmax[j] * img.shape[1]))
        ymax = int(round(top_ymax[j] * img.shape[0]))
        score = top_conf[j]
        label = int(top_label_indices[j])
        label_name = voc_classes[label - 1]

        if label_name in feeder_list:
            resolution = (ymax-ymin)*(xmax-xmin)

            #if the detected object is bigger than 400 pixels (e.g. 20 x 20)
            if resolution >= 400:
                feeder_resolution = np.append(feeder_resolution, resolution)

                cropped_img = img[ymin:ymax, xmin:xmax, :]
                feeder_in_single_image.append(cropped_img)

    if len(feeder_in_single_image) > 0:
        max_resolution_index = np.argmax(feeder_resolution)
        #plt.imshow(feeder_in_single_image[max_resolution_index-1] / 255.)
        #plt.show();
        ssd_img = np.uint8(feeder_in_single_image[max_resolution_index-1])
    else:
        ssd_img = img

    return ssd_img

def predict_img(img_name):
    img_original = mpimg.imread(img_name)
    fig, ax = plt.subplots()
    #ax.imshow(img_original)
    #plt.show()

    inputs = []
    images = []

    img = image.load_img(img_name, target_size=(300, 300))
    img = image.img_to_array(img)
    images.append(imread(img_name))
    inputs.append(img.copy())
    inputs = preprocess_input(np.array(inputs))
    preds = model.predict(inputs, batch_size=1, verbose=0)
    results = bbox_util.detection_out(preds)

    for i, img in enumerate(images):
        if type(results[i]) is not list:
            ssd_img = ssd_image(img, results, i)
            
            ax.imshow(ssd_img)
            plt.show()
            
            resize_img = imresize(ssd_img, (img_size, img_size))

            x = np.expand_dims(resize_img, axis=0)
            y_pred = model_cnn.predict(x)
            if y_pred < 0.6:
                print(y_pred[0][0], 'not empty')
            else:
                print(y_pred[0][0], 'empty')
        else:
            ax.imshow(img_original)
            plt.show()
            
            y_pred = 0.00
            print('0.00 not empty')

predict_img('./data/v2/test/0/12.jpg')

predict_img('./data/v2/test/1/12.jpg')

predict_img('./data/v2/test/0/31.jpg')

import random

predict_img('./data/v2/test/0/'+str(int(random.random() * num_test + 1))+'.jpg')

import random

predict_img('./data/v2/test/1/'+str(int(random.random() * num_test + 1))+'.jpg')

true_positive = 0
false_negative = 0

for i in range(num_test):
    inputs = []
    images = []

    img_name = './data/v1/test/1/'+str(i + 1)+'.jpg'
    img = image.load_img(img_name, target_size=(300, 300))
    img = image.img_to_array(img)
    images.append(imread(img_name))
    inputs.append(img.copy())
    inputs = preprocess_input(np.array(inputs))
    preds = model.predict(inputs, batch_size=1, verbose=0)
    results = bbox_util.detection_out(preds)

    for i, img in enumerate(images):
        if type(results[i]) is not list:
            ssd_img = ssd_image(img, results, i)
            resize_img = imresize(ssd_img, (img_size, img_size))

            x = np.expand_dims(resize_img, axis=0)
            y_pred = model_cnn.predict(x)
        else:
            y_pred = 0.00
    
    if y_pred > 0.6:
        true_positive = true_positive + 1
    else:
        false_negative = false_negative + 1
        
        #Show the image
        fig, ax = plt.subplots()
        ax.imshow(img)
        print(y_pred)
        #plt.show()

print("True positive: " + str(true_positive))
print("False negative: " + str(false_negative))

true_negative = 0
false_positive = 0

for i in range(num_test):
    inputs = []
    images = []

    img_name = './data/v1/test/0/'+str(i + 1)+'.jpg'
    img = image.load_img(img_name, target_size=(300, 300))
    img = image.img_to_array(img)
    images.append(imread(img_name))
    inputs.append(img.copy())
    inputs = preprocess_input(np.array(inputs))
    preds = model.predict(inputs, batch_size=1, verbose=0)
    results = bbox_util.detection_out(preds)

    for i, img in enumerate(images):
        if type(results[i]) is not list:
            ssd_img = ssd_image(img, results, i)
            resize_img = imresize(ssd_img, (img_size, img_size))

            x = np.expand_dims(resize_img, axis=0)
            y_pred = model_cnn.predict(x)
        else:
            y_pred = 0.00
    
    if y_pred > 0.6:
        false_positive = false_positive + 1
        
        #Show the image
        fig, ax = plt.subplots()
        ax.imshow(img)
        print(y_pred)
        #plt.show()

    else:
        true_negative = true_negative + 1

print("True negative: " + str(true_negative))
print("False positive: " + str(false_positive))

print("Recall: " + str(true_positive / (true_positive + false_negative)))
print("Precision: " + str(true_positive / (true_positive + false_positive)))

recall_list = []
precision_list = []

for i in range(1, 10):

    confidence = i/10

    true_positive = 0
    false_negative = 0
    true_negative = 0
    false_positive = 0

    for j in range(num_test):
        inputs = []
        images = []
    
        img_name = './data/v1/test/0/'+str(j + 1)+'.jpg'
        img = image.load_img(img_name, target_size=(300, 300))
        img = image.img_to_array(img)
        images.append(imread(img_name))
        inputs.append(img.copy())
        inputs = preprocess_input(np.array(inputs))
        preds = model.predict(inputs, batch_size=1, verbose=0)
        results = bbox_util.detection_out(preds)

        for i, img in enumerate(images):
            if type(results[i]) is not list:
                ssd_img = ssd_image(img, results, i)
                resize_img = imresize(ssd_img, (img_size, img_size))

                x = np.expand_dims(resize_img, axis=0)
                y_pred = model_cnn.predict(x)
            else:
                y_pred = 0.00

        if y_pred < confidence:
            true_positive = true_positive + 1
        else:
            false_negative = false_negative + 1

        inputs2 = []
        images2 = []
        
        img_name2 = './data/v1/test/1/'+str(j + 1)+'.jpg'
        img2 = image.load_img(img_name2, target_size=(300, 300))
        img2 = image.img_to_array(img2)
        images2.append(imread(img_name2))
        inputs2.append(img2.copy())
        inputs2 = preprocess_input(np.array(inputs2))
        preds2 = model.predict(inputs2, batch_size=1, verbose=0)
        results2 = bbox_util.detection_out(preds2)
        
        for i, img in enumerate(images2):
            if type(results2[i]) is not list:
                ssd_img2 = ssd_image(img, results2, i)
                resize_img2 = imresize(ssd_img2, (img_size, img_size))

                x2 = np.expand_dims(resize_img2, axis=0)
                y_pred2 = model_cnn.predict(x2)
            else:
                y_pred2 = 0.00

        if y_pred2 < confidence:
            false_positive = false_positive + 1
        else:
            true_negative = true_negative + 1

    recall_list.append(true_positive / (true_positive + false_negative))
    precision_list.append(true_positive / (true_positive + false_positive))

confidence_level = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

plt.plot(confidence_level, recall_list)
plt.plot(confidence_level, precision_list)
plt.title('Recall (orange) and Precision (blue)')
plt.show()



