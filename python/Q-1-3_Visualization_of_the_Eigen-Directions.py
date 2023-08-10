import sys
print(sys.version)

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import time

import pandas as pd
import seaborn as sns

import sys
sys.path.append('../code/')

from pca import Pca, make_image

from classification_base import MNIST_PATH
from mnist_helpers import mnist_training, mnist_testing

train_X, train_y = mnist_training(shuffled=True)
#test_X, test_y = mnist_testing(shuffled=True)

eigenvectors = np.load("./data/Q-1-2_eigenvectors.npy")
sigma = np.load("./data/Q-1-2_sigma.npy")

# Show two native numbers.
make_image(train_X[1,:])
make_image(train_X[2,:])

from sklearn.decomposition import PCA
pca = PCA(n_components=10)
pca.fit(train_X)
make_image(pca.components_[1,:])

make_image(eigenvectors[:,0])

make_image(eigenvectors[:,1])

# Loop over first 10 eigenvectors and save them.
for i in range(0, 10):
    make_image(eigenvectors[:,i], "../figures/eigenvectors/ev_{}.png".format(i))
    make_image(eigenvectors[:,i], "ev_{}.png".format(i))

# convert +append a.png b.png c.png
get_ipython().system('convert +append ev_0.png ev_1.png ev_2.png ev_3.png ev_4.png ev_5.png ev_6.png ev_7.png ev_8.png ev_9.png ../figures/eigenvectors.jpg ')

get_ipython().system(' rm ev_*.png')



