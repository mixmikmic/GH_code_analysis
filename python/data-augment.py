from IPython.display import display
import numpy as np
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from ipywidgets import interact, fixed
import ipywidgets as widgets

img = load_img('images/cat001.jpg')
x = img_to_array(img)
img

