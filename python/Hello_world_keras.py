get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123)

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils

batch_size = 128
nb_classes = 10

img_rows,img_cols = 28,28

(x_train,y_train),(x_test,y_test) = mnist.load_data('/home/murugesan/DEV/Dataset/mnist.pkl')



