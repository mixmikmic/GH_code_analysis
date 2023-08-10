# read in the geotiff with rasterio
# copy the channel as 3 channels, 
# make a png

import rasterio
import matplotlib.pyplot as plt 
import scipy.misc
import numpy as np


count = 0

root_folder = '/home/ubuntu/data/sar/train/train_3classes_240/other/'
for filename in os.listdir (root_folder):
    data_path = os.path.join(root_folder, filename)

    with rasterio.open(data_path) as raster:
        img_array = raster.read(1)
    count +=1
#     plt.imshow(img_array)
#     plt.show()
    png_path = data_path.replace('.tif', '.png')
    scipy.misc.imsave(png_path, img_array)
    
print(count)

# # used to make the validation data - commented so do not run again. 

import glob
from shutil import move
from shutil import copyfile

get_ipython().magic('cd /home/ubuntu/data/sar/train/train_3classes_140/oil_platform/')
g = glob.glob('*.png')
shuf = np.random.permutation(g)
for i in range(30): copyfile(shuf[i], '/home/ubuntu/data/sar/train/sample_240/valid/' +'/turbine/' + shuf[i])
# for i in range(200): move(shuf[i], '/home/ubuntu/data/sar/train/valid_3classes_140' +'/oil_platform/' + shuf[i])

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Flatten
import numpy as np
from keras.utils.np_utils import to_categorical

base_model = VGG16(weights='imagenet', include_top=False)

for l in base_model.layers:
    l.trainable =False
base_model.summary()

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
#     rescale=1./255)
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
        '/home/ubuntu/data/sar/train/train_3classes_240',
        target_size=(224, 224),
        batch_size=10,
        class_mode='categorical')


validation_generator = test_datagen.flow_from_directory(
        '/home/ubuntu/data/sar/train/valid_3classes_240', 
        target_size=(224, 224),
        shuffle=False,
        batch_size=10,
        class_mode='categorical')

train_generator.next.im_self.color_mode

import cv2

img = cv2.imread('/home/ubuntu/data/sar/train/valid_3classes_240/other/S1A_IW_GRDH_1SDV_20170214T062124_20170214T062149_015276_019087_AF6B_terrain_correction_2.png')

plt.imshow(img)
plt.show()

img.shape

from keras.preprocessing.image import ImageDataGenerator
def preprocess_input(x, data_format=None):
    """Preprocesses a tensor encoding a batch of images.
    # Arguments
        x: input Numpy tensor, 4D.
        data_format: data format of the image tensor.
    # Returns
        Preprocessed tensor.
    """

    # 'RGB'->'BGR'
    x = x[:, :, ::-1]
    # Zero-center by mean pixel
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    return x


def create_precomputed_data(model, batches):
    filenames = batches.filenames
    conv_features = model.predict_generator(batches, (batches.samples / batches.batch_size), verbose=1)
    labels_onehot = to_categorical(batches.classes)
    labels = batches.classes
    return (filenames, conv_features, labels_onehot, labels)


trn_filenames, trn_conv_features, trn_labels, trn_labels_1 = create_precomputed_data(base_model, train_generator)
val_filenames, val_conv_features, val_labels, val_labels_1 = create_precomputed_data(base_model, validation_generator)

trn_conv_features.shape

RESULTS_DIR = '/home/ubuntu/data/sar/train/results'

import bcolz
def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def save_precomputed_data(filenames, conv_feats, labels, features_base_name="VGG240crops_conv_feats/trn_"):
    save_array(RESULTS_DIR+"/"+features_base_name+'filenames.dat', np.array(filenames))
    save_array(RESULTS_DIR+"/"+features_base_name+'conv_feats.dat', conv_feats)
    save_array(RESULTS_DIR+"/"+features_base_name+'labels.dat', np.array(labels))
    
save_precomputed_data(trn_filenames, trn_conv_features, trn_labels, "VGG240crops_conv_feats/trn_")
save_precomputed_data(val_filenames, val_conv_features, val_labels, "VGG240crops_conv_feats/val_")


import bcolz
def load_array(fname):
    return bcolz.open(fname)[:]

def load_precomputed_data(features_base_name="VGG240crops_conv_feats/trn_"):
    filenames = load_array(RESULTS_DIR+"/"+features_base_name+'filenames.dat').tolist()
    conv_feats = load_array(RESULTS_DIR+"/"+features_base_name+'conv_feats.dat')
    labels = load_array(RESULTS_DIR+"/"+features_base_name+'labels.dat')
    return filenames, conv_feats, labels

trn_filenames, trn_conv_features, trn_labels = load_precomputed_data("VGG240crops_conv_feats/trn_")
val_filenames, val_conv_features, val_labels = load_precomputed_data("VGG240crops_conv_feats/val_")


from keras.layers import Input, BatchNormalization, Dense, Dropout, Conv2D
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, Activation
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from datetime import datetime
import distutils.dir_util
from keras.callbacks import CSVLogger

p=0.7
classifier_input_shape = (7, 7, 512)
# classifier_input_shape = resnet_base.layers[-1].output_shape[1:]
classifier_input = Input(shape=classifier_input_shape)

# Create classifier model
      
x = Flatten()(classifier_input)
x = BatchNormalization(axis=1)(x)
x = Dropout(p/4)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(p/2)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(p)(x)
x = Dense(3, activation='softmax')(x)
                                                     
classifier_model_v1 = Model(classifier_input, x)

#from keras.optimizers import SGD
classifier_model_v1.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


classifier_model_v1.summary()

def fit_precomputed_helper(model, result_dir_name, lr=0.1, nb_epoch=1):  
    K.set_value(model.optimizer.lr, lr)
    
    now = datetime.now().strftime("%Y%m%d_%H%M%S.h5")
    results_dir = RESULTS_DIR + "/" + result_dir_name + "/"
    distutils.dir_util.mkpath(results_dir)
    
    model.fit(trn_conv_features, trn_labels,
              batch_size=32, 
              epochs=nb_epoch,
              validation_data=(val_conv_features, val_labels),
              shuffle=True, 
              callbacks=[CSVLogger(results_dir+"epoch_results.csv", separator=',', append=True)])
    model.save_weights(results_dir + now)
    return model

classifier_model_v1 = fit_precomputed_helper(classifier_model_v1, "classifier_model_v1", lr=0.0001, nb_epoch=10)

classifier_model_v1.save_weights('/home/ubuntu/git/learningWithKaggle/weights/fishing/ft_resnet93_valid.h5')

nf = 128
p = 0.4

# x = Flatten(input_shape=classifier_input_shape)(classifier_input)
x = Conv2D(nf,(3,3), activation='relu', padding='same')(classifier_input)
x = BatchNormalization(axis=1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(nf,(3,3), activation='relu', padding='same')(x)
x = BatchNormalization(axis=1)(x)
x = MaxPooling2D()(x)
x = Conv2D(nf,(3,3), activation='relu', padding='same')(x)
x = BatchNormalization(axis=1)(x)
# x = MaxPooling2D((1,2))(x)
x = Conv2D(3,(3,3), padding='same')(x)
x = Dropout(p)(x)
x = GlobalAveragePooling2D()(x)
x = Activation('softmax')(x)

classifier_model_v2 = Model(classifier_input, x)

classifier_model_v2.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

classifier_model_v2 = fit_precomputed_helper(classifier_model_v2, "classifier_model_v2", lr=0.0001, nb_epoch=10)

# Create classifier model

x= Flatten()(classifier_input)
x = Dense(3, activation='softmax')(x)
                                                     
classifier_model_v3 = Model(classifier_input, x)

#from keras.optimizers import SGD
classifier_model_v3.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

classifier_model_v3 = fit_precomputed_helper(classifier_model_v3, "classifier_model_v3", lr=0.001, nb_epoch=50)



