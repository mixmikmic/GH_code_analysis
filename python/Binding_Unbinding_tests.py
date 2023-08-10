import tensorflow as tf
import random
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')

#Circular Convolution Definition
#adapted from here https://github.com/fchollet/keras/issues/2518
def holographic_merge(x,y):
    """
    Computes the 1d discrete circular convolution of two vectors x,y
    """
    x_fft = tf.fft(tf.complex(x, 0.0))
    y_fft = tf.fft(tf.complex(y, 0.0))
    ifft = tf.ifft(tf.conj(x_fft) * y_fft)
    return tf.cast(tf.real(ifft), 'float32')

def holographic_merge_2d(x,y):
    """
    Computes the 2d discrete circular convolution of two matrices x,y
    """
    x_fft = tf.fft2d(tf.complex(x, 0.0))
    y_fft = tf.fft2d(tf.complex(y, 0.0))
    ifft = tf.ifft2d(tf.conj(x_fft) * y_fft)
    return tf.cast(tf.real(ifft), 'float32')

#Need to do something to calculate the inverse circular convolution.
#for this, according to:
# How to Build a Brain -> D.2 Learning High-Level Transformations, Eliasmith 2013
#we can compute the involution which is:
# x = [x0, x1, x2, ... xd-2, xd-1] for dimension d
# x' = [x0, xd-1, xd-2, .... x2, x1] which can be reasoned in python as:
# x1 = x[0]+x[1:].reverse()

#Or we can define a permutation matrix S such as Sx=x' and based on this:
# x=z(*)y' = ifft( fft(z) . (fft(S)y))

#TODO make this more efficient with a cache
cache_S = {}

def get_S(x):
    """
    Computes a matrix that allows for the permutation 
    x = [x0, x1, x2, ... xd-2, xd-1] for dimension d
    x' = [x0, xd-1, xd-2, .... x2, x1] which can be reasoned in python as:
    x1 = x[0]+x[1:].reverse()
    The matrix has shape:
    [
    [1, 0, 0, 0, ...., 0, 0, 0]
    [0, 0, 0, 0, ...., 0, 0, 1]
    [0, 0, 0, 0, ...., 0, 1, 0]
    [0, 0, 0, 0, ...., 1, 0, 0]
    [ ....................... ]
    [0, 0, 0, 1, ...., 0, 0, 0]
    [0, 0, 1, 0, ...., 0, 0, 0]
    [0, 1, 0, 0, ...., 0, 0, 0]
    ]
    """
    #create a diagonal matrix
    sp = tf.shape(x) #input vector shape
    dim = x.get_shape().num_elements()
    #lookout in the S matrix cache if exists
    if (dim in cache_S):
        return cache_S[dim]
    #TODO make this faster
    permut = [0] + [i+1 for i in range(dim)][::-1] #permutation needed to create the x' 
    d = tf.diag(tf.ones(sp, tf.int32))
    S = tf.transpose(d, perm=permut)
    return S

def holographic_unmerge(z,y):
    """
    Computes the 1d discrete circular DEconvolution of two vectors z,y
    """
    z_fft = tf.fft(tf.complex(z, 0.0))
    #compute the S matrix .... for approximating the inverse y' (t_inv) of y
    S = get_S(y)
    s_fft2d = tf.fft2d(tf.complex(S, 0.0))
    y_inv = tf.tensordot(s_fft2d,y)
    ifft = tf.ifft(tf.conj(z_fft) * y_inv)
    return tf.cast(tf.real(ifft), 'float32')

#TODO make the 2D transformation .. is a bit harder to make it

#issue with TensorFlow, reference here: https://github.com/tensorflow/tensorflow/issues/6698
#config needed
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)

def rand_sparse_vector():
    initial = tf.concat([tf.random_normal([10]),tf.zeros(100)],axis=0)
    return tf.Variable(initial)

x1 = rand_sparse_vector()

y1 = rand_sparse_vector()

z1 = holographic_merge(x1,y1)

z1

x1p = holographic_unmerge(z1,y1)



