from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

#PIL.Image.open can read an image from disk
im=Image.open("luna.jpg")

#and pyplot's imshow works great to display an image in a notebook.

plt.imshow(im)

im = im.resize((256,256))
plt.imshow(im)

cat_array = np.array(im)

cat_array.shape

#if we select only 
cat_array[:,:,0]

cat_array[:,:,0].shape

plt.imshow(cat_array[:,:,0])

plt.imshow(cat_array)







