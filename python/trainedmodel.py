
from scipy.misc import imsave
from keras import metrics
import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')
import importlib

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions, preprocess_input

import math, keras, datetime, pandas as pd, numpy as np, keras.backend as K
import tarfile, tensorflow as tf, matplotlib.pyplot as plt, xgboost, operator, random, pickle, glob, os
import shutil, sklearn, functools, itertools, scipy
from PIL import Image
import cv2
import tqdm
import tarfile

##Reading all the image names in our training data
##Change the path to the location of your own dataset
path='train/'

img_list = glob.glob(path+'**/*.JPEG', recursive=True)
n = len(img_list); n


##Lambda funtion for preprocessing input image because we are using VGG16
##We have to subtract mean from each channel and also convert RGB->BGR
rn_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
preproc = lambda x: (x - rn_mean)[:, :, :, ::-1]


def read_image(img_path,H,W):
    #Reading Image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    #converting RGB-->BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #Resize the image
    img = cv2.resize(img, (W,H)) 
    return img

#Converting images to numpy array
arr_lr = np.array(image_lr)
arr_hr = np.array(image_hr)

#Reading images as both low resolution and high resolution
image_lr=[]
image_hr=[]

TRAIN_PATH='/train/*'

for img in img_list:
    image_lr.append(read_image(img,72,72))
    
for img in img_list:
    image_hr.append(read_image(img,288,288))

#low resolution images
print(arr_lr.shape)
#high resolution images
print(arr_hr.shape)

plt.imshow(arr_lr[7])

plt.imshow(arr_hr[7])

def conv_block(x, filters, size, stride=(2,2), mode='same', act=True):
    x = Convolution2D(filters, size, size, subsample=stride, border_mode=mode)(x)
    x = BatchNormalization(mode=0)(x)
    return Activation('relu')(x) if act else x

def res_block(ip, nf=64):
    x = conv_block(ip, nf, 3, (1,1))
    x = conv_block(x, nf, 3, (1,1), act=False)
    return merge([x, ip], mode='sum')

def deconv_block(x, filters, size, shape, stride=(2,2)):
    x = Deconvolution2D(filters, size, size, subsample=stride, 
        border_mode='same', output_shape=(None,)+shape)(x)
    x = BatchNormalization(mode=0)(x)
    return Activation('relu')(x)

def up_block(x, filters, size):
    x = keras.layers.UpSampling2D()(x)
    x = Convolution2D(filters, size, size, border_mode='same')(x)
    x = BatchNormalization(mode=0)(x)
    return Activation('relu')(x)

arr_lr.shape


##This is the model we are interested in training
##It take (72,72,3) image and convert it into (288,288,3)

#Creating Input tensor of (72,72,3)
#This will be our Input to the model
inp=Input(arr_lr.shape[1:])
#Adding convolution block that we defined above
x=conv_block(inp, 64, 9, (1,1))
#Adding res_block
for i in range(4): x=res_block(x)
#Upsampling the model
x=up_block(x, 64, 3)
#Another layer of Upsampling
x=up_block(x, 64, 3)
#Final conolution layer
x=Convolution2D(3, 9, 9, activation='tanh', border_mode='same')(x)
outp=Lambda(lambda x: (x+1)*127.5)(x)

#This is how the above model looks
model=Model(inp,outp)
model.summary()

vgg_inp=Input(arr_hr.shape[1:])

vgg_inp.shape

#We will this VGG16 model only to calculate our perceptual loss so we will make each layer untrainable
#In this model we will extract model upto a layer (block2_conv2) and then pass our output above above model and actual
#image of (288,288,3) and compare their activation at block2_conv2 to calculate loss
vgg= VGG16(include_top=False, input_tensor=Lambda(preproc)(vgg_inp))
for l in vgg.layers: l.trainable=False

vgg.summary()



# getting the layer
vgg_out_layer = vgg.get_layer('block2_conv2').output

# making model Model(inputs, outputs)
vgg_content = Model(vgg_inp, vgg_out_layer)

vgg_content.summary()




# passing actual high resolution image to our network
vgg_hr_image = vgg_content(vgg_inp)

# passing predicted high resolution image to our network
vgg_it_op = vgg_content(outp)


vgg_hr_image.shape

vgg_it_op.shape

#This is our perceptual loss function
loss = Lambda(lambda x: K.sqrt(K.mean((x[0]-x[1])**2, (1,2))))([vgg_hr_image, vgg_it_op])

#Final model
sr_model = Model([inp, vgg_inp], loss)
sr_model.compile('adam', 'mse')

#After calculating the loss function this model model will out a tensor of (None,128)
sr_model.summary()

#This is where we store our (None,128) out
targ = np.zeros((arr_hr.shape[0], 128))
targ.shape

#setting learning rate
K.set_value(sr_model.optimizer.lr, 1e-4)
sr_model.fit([arr_lr, arr_hr], targ, 16, 1)



K.set_value(sr_model.optimizer.lr, 1e-3)
sr_model.fit([arr_lr, arr_hr], targ, 16, 1)

top_model = Model(inp, outp)

top_model.summary()

p = top_model.predict(arr_lr[10:11])

plt.imshow(arr_lr[10].astype('uint8'));

#Predicted high resolution image
plt.imshow(p[0].astype('uint8'));

plt.imshow(arr_hr[10].astype('uint8'));



top_model.save_weights('sr_final.h5')



sr_model.save_weights('sr_main_model.h5')

from IPython.display import FileLink, FileLinks
FileLink('sr_main_model.h5')

