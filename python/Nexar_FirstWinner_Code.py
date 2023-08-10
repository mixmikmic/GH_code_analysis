import os
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

caffe_root = '/home/bhushan/caffe-master/' 
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

model_def = "./model/deploy.prototxt"
model_weights = "./model/train_squeezenet_scratch_trainval_manual_p2__iter_8000.caffemodel"

def class_idx_to_name(idx):
    return ['none', 'red', 'green'][idx]

from caffe.classifier import Classifier

c = Classifier(
           model_def, 
           model_weights, 
           mean=np.array([104, 117, 123]),
           raw_scale=255,
           channel_swap=(2,1,0),
           image_dims=(256, 256)
)

# set batch size
BATCH_SIZE = 1
c.blobs['data'].reshape(BATCH_SIZE, 3, c.blobs['data'].shape[2], c.blobs['data'].shape[3])
c.blobs['prob'].reshape(BATCH_SIZE, 3)
c.reshape()

import os, random

images_path = './Test_Images/Test4.jpg'
image = caffe.io.load_image(images_path)
cls = c.predict([image]).argmax()
plt.imshow(image)
plt.axis('off')
print 'predicted class is:', class_idx_to_name(cls)



