import sys
print(sys.version)

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import subprocess
import time

import pandas as pd
import seaborn as sns

import sys
sys.path.append('../code/')

from pca import Pca, make_image

from classification_base import MNIST_PATH
from mnist_helpers import mnist_training, mnist_testing

X_orig = np.load("./data/digits_0_through_4--untransformed.npy")

X_2_components = np.load("./data/digits_0_through_4_transformed--2_components.npy")
X_5_components = np.load("./data/digits_0_through_4_transformed--5_components.npy")
X_10_components = np.load("./data/digits_0_through_4_transformed--10_components.npy")
X_20_components = np.load("./data/digits_0_through_4_transformed--20_components.npy")
X_50_components = np.load("./data/digits_0_through_4_transformed--50_components.npy")

X_50_components.shape

make_image(X_2_components[2])

make_image(X_50_components[2])

X_50_components[2].shape

d = {"original": X_orig,
     2:X_2_components, 
     5:X_5_components,   
     10:X_10_components, 
     20:X_20_components, 
     50:X_50_components}

subprocess.call("date")

subprocess.call("ls")

get_ipython().system(' mkdir ../figures/PCA_reconstructions')

# Loop over first 10 eigenvectors and save them.
for k, images in d.items():
    #print(k)
    #print(images.shape)
    for i in range(images.shape[0]):
        image = images[i]
        #print(image.shape)
        #dir = "./figures/PCA_images/"
        path="{}_c--i_{}.png".format(k,i)
        #print(path)
        make_image(image, path=path)
    shell_stitch_command = "convert +append {0}_c--i_0.png {0}_c--i_1.png {0}_c--i_2.png {0}_c--i_3.png {0}_c--i_4.png {0}_stitched.png ".format(k)
    print(shell_stitch_command)
    subprocess.call(shell_stitch_command, shell=True)
    subprocess.call("""mv *.png ../figures/PCA_reconstructions/""", shell=True)
    #make_image(eigenvectors[:,i], "ev_{}.png".format(i))

from IPython.display import Image

PATH = "../figures/PCA_reconstructions/"
Image(filename = PATH + "2_stitched.png", width=500, height=100)

Image(filename = PATH + "5_stitched.png", width=500, height=100)

Image(filename = PATH + "10_stitched.png", width=500, height=100)

Image(filename = PATH + "20_stitched.png", width=500, height=100)

Image(filename = PATH + "50_stitched.png", width=500, height=100)

# Check the rest

all_X_transformed_50 = np.load("./data/X_transformed_by_50_components.npy")

all_X_transformed_50.shape

