import os
import copy 

import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
from keras.models import load_model
from keras.utils.generic_utils import get_custom_objects
from keras.backend import binary_crossentropy
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

smooth = 1e-12

def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)

def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_loss(y_true, y_pred):
    return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)

weight_path = '/root/data/hackathon/thomas_augmentation_weights_16.hdf5' 
model = load_model(weight_path, custom_objects={'jaccard_coef_loss': jaccard_coef_loss, 
                                                'jaccard_coef_int': jaccard_coef_int})

before = '/root/thomas/github/stanfordHacks/presentation/syria/before.png'
after = '/root/thomas/github/stanfordHacks/presentation/syria/after.png'

crop = np.array(Image.open(before))[:256, -256:]
plt.figure(figsize=(15, 15))
plt.imshow(crop)
plt.show()

out = model.predict(np.expand_dims(crop, axis=0))

img = Image.fromarray(crop)
mask = np.zeros((256, 256, 3), dtype=np.uint8)
tmp = copy.deepcopy(out[0,...,0])
tmp[tmp>0.1] = 255
mask[:,:,0] = tmp
mask = Image.fromarray(mask)

Image.blend(img, mask, 0.5)



