#Grab inception model from online and unzip it (you can skip this step if you've already downloaded the model.
get_ipython().system('wget -P ../data/ https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip')
get_ipython().system('unzip ../data/inception5h.zip -d ../data/inception5h/')
get_ipython().system('rm ../data/inception5h.zip')

from __future__ import print_function
from io import BytesIO
import math, time, copy, json, os
import glob
from os import listdir
from os.path import isfile, join
from random import random
from io import BytesIO
from enum import Enum
from functools import partial
import PIL.Image
from IPython.display import clear_output, Image, display, HTML
import numpy as np
import scipy.misc
import tensorflow as tf
from lapnorm import *

for l, layer in enumerate(layers):
    layer = layer.split("/")[1]
    num_channels = T(layer).shape[3]
    print(layer, num_channels)

def render_naive(t_obj, img0, iter_n=20, step=1.0):
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
    img = img0.copy()
    for i in range(iter_n):
        g, score = sess.run([t_grad, t_score], {t_input:img})
        # normalizing the gradient, so the same step size should work 
        g /= g.std()+1e-8         # for different layers and networks
        img += g*step
    return img

img0 = np.random.uniform(size=(200, 200, 3)) + 100.0
layer = 'mixed4d_5x5_bottleneck_pre_relu'
channel = 20
img1 = render_naive(T(layer)[:,:,:,channel], img0, 40, 1.0)
display_image(img1)

def render_multiscale(t_obj, img0, iter_n=10, step=1.0, octave_n=3, octave_scale=1.4):
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
    img = img0.copy()
    for octave in range(octave_n):
        if octave>0:
            hw = np.float32(img.shape[:2])*octave_scale
            img = resize(img, np.int32(hw))
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            # normalizing the gradient, so the same step size should work 
            g /= g.std()+1e-8        # for different layers and networks
            img += g*step
        print("octave %d/%d"%(octave+1, octave_n))
    clear_output()
    return img

h, w = 200, 200
octave_n = 3
octave_scale = 1.4
iter_n = 30

img0 = np.random.uniform(size=(h, w, 3)) + 100.0

layer = 'mixed4d_5x5_bottleneck_pre_relu'
channel = 25

img1 = render_multiscale(T(layer)[:,:,:,channel], img0, iter_n, 1.0, octave_n, octave_scale)
display_image(img1)

h, w = 240, 240
octave_n = 3
octave_scale = 1.4
iter_n = 30

img0 = load_image('../assets/kitty.jpg', h, w)

layer = 'mixed4d_5x5_bottleneck_pre_relu'
channel = 21

img1 = render_multiscale(T(layer)[:,:,:,channel], img0, iter_n, 1.0, octave_n, octave_scale)
display_image(img1)

def render_lapnorm(t_obj, img0, iter_n=10, step=1.0, oct_n=3, oct_s=1.4, lap_n=4):
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
    # build the laplacian normalization graph
    lap_norm_func = tffunc(np.float32)(partial(lap_normalize, scale_n=lap_n))
    img = img0.copy()
    for octave in range(oct_n):
        if octave>0:
            hw = np.float32(img.shape[:2])*oct_s
            img = resize(img, np.int32(hw))
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            g = lap_norm_func(g)
            img += g*step
            print('.', end='')
        print("octave %d/%d"%(octave+1, oct_n))
    clear_output()
    return img
 

h, w = 300, 400
octave_n = 3
octave_scale = 1.4
iter_n = 10

img0 = np.random.uniform(size=(h, w, 3)) + 100.0

layer = 'mixed4d_5x5_bottleneck_pre_relu'
channel = 25

img1 = render_lapnorm(T(layer)[:,:,:,channel], img0, iter_n, 1.0, octave_n, octave_scale)
display_image(img1)

def lapnorm_multi(t_obj, img0, mask, iter_n=10, step=1.0, oct_n=3, oct_s=1.4, lap_n=4, clear=True):
    mask_sizes = get_mask_sizes(mask.shape[0:2], oct_n, oct_s)
    img0 = resize(img0, np.int32(mask_sizes[0])) 
    t_score = [tf.reduce_mean(t) for t in t_obj] # defining the optimization objective
    t_grad = [tf.gradients(t, t_input)[0] for t in t_score] # behold the power of automatic differentiation!
    # build the laplacian normalization graph
    lap_norm_func = tffunc(np.float32)(partial(lap_normalize, scale_n=lap_n))
    img = img0.copy()
    for octave in range(oct_n):
        if octave>0:
            hw = mask_sizes[octave] #np.float32(img.shape[:2])*oct_s
            img = resize(img, np.int32(hw))
        oct_mask = resize(mask, np.int32(mask_sizes[octave]))
        for i in range(iter_n):
            g_tiled = [lap_norm_func(calc_grad_tiled(img, t)) for t in t_grad]
            for g, gt in enumerate(g_tiled):
                img += gt * step * oct_mask[:,:,g].reshape((oct_mask.shape[0],oct_mask.shape[1],1))
            print('.', end='')
        print("octave %d/%d"%(octave+1, oct_n))
    if clear:
        clear_output()
    return img

h, w = 300, 400
octave_n = 3
octave_scale = 1.4
iter_n = 10

img0 = np.random.uniform(size=(h, w, 3)) + 100.0

objectives = [T('mixed3a_3x3_bottleneck_pre_relu')[:,:,:,25], 
              T('mixed4d_5x5_bottleneck_pre_relu')[:,:,:,15]]

# mask
mask = np.zeros((h, w, 2))
mask[:150,:,0] = 1.0
mask[150:,:,1] = 1.0

img1 = lapnorm_multi(objectives, img0, mask, iter_n, 1.0, octave_n, octave_scale)
display_image(img1)

h, w = 400, 400
octave_n = 3
octave_scale = 1.4
iter_n = 10

img0 = load_image('../assets/kitty.jpg', h, w)

objectives = [T('mixed4d_3x3_bottleneck_pre_relu')[:,:,:,125], 
              T('mixed5a_5x5_bottleneck_pre_relu')[:,:,:,30]]

# mask
mask = np.zeros((h, w, 2))
mask[:,:200,0] = 1.0
mask[:,200:,1] = 1.0

img1 = lapnorm_multi(objectives, img0, mask, iter_n, 1.0, octave_n, octave_scale)
display_image(img1)

h, w = 256, 1024

img0 = np.random.uniform(size=(h, w, 3)) + 100.0

octave_n = 3
octave_scale = 1.4
objectives = [T('mixed3b_5x5_bottleneck_pre_relu')[:,:,:,9], 
              T('mixed4d_5x5_bottleneck_pre_relu')[:,:,:,17]]

mask = np.zeros((h, w, 2))
mask[:,:,0] = np.linspace(0,1,w)
mask[:,:,1] = np.linspace(1,0,w)

img1 = lapnorm_multi(objectives, img0, mask, iter_n=20, step=1.0, oct_n=3, oct_s=1.4, lap_n=4)

print("image")
display_image(img1)
print("masks")
display_image(255*mask[:,:,0])
display_image(255*mask[:,:,1])

h, w = 500, 500

cy, cx = 0.5, 0.5

# circle masks
pts = np.array([[[i/(h-1.0),j/(w-1.0)] for j in range(w)] for i in range(h)])

ctr = np.array([[[cy, cx] for j in range(w)] for i in range(h)])


pts -= ctr
dist = (pts[:,:,0]**2 + pts[:,:,1]**2)**0.5
dist = dist / np.max(dist)



mask = np.ones((h, w, 2))
mask[:, :, 0] = dist
mask[:, :, 1] = 1.0-dist


img0 = np.random.uniform(size=(h, w, 3)) + 100.0

octave_n = 3
octave_scale = 1.4
objectives = [T('mixed3b_5x5_bottleneck_pre_relu')[:,:,:,9], 
              T('mixed4d_5x5_bottleneck_pre_relu')[:,:,:,17]]


img1 = lapnorm_multi(objectives, img0, mask, iter_n=20, step=1.0, oct_n=3, oct_s=1.4, lap_n=4)
display_image(img1)

h, w = 200, 200

# start with random noise
img = np.random.uniform(size=(h, w, 3)) + 100.0

octave_n = 3
octave_scale = 1.4
objectives = [T('mixed4d_5x5_bottleneck_pre_relu')[:,:,:,11]]
mask = np.ones((h, w, 1))

# repeat the generation loop 20 times. notice the feedback -- we make img and then use it the initial input 
for f in range(20):
    img = lapnorm_multi(objectives, img, mask, iter_n=20, step=1.0, oct_n=3, oct_s=1.4, lap_n=4, clear=False)
    display_image(img)    # let's see it
    img = resize(img[10:-10,10:-10,:], (h, w))  # before looping back, crop the border by 10 pixels, resize, repeat

