
import cv2
from PIL import Image

#%matplotlib inline
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
#import mpld3
#mpld3.enable_notebook()

import os,inspect
import sys 

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from grid_distortion import warp_image
from howe import binarize



def rgb(img):
    b,g,r = cv2.split(img)
    rgb_img = cv2.merge([r,g,b])
    return rgb_img

def to_grayscale(img):
    image_reshape = np.swapaxes(img, 0, 2)
    image_reshape = np.swapaxes(image_reshape, 0, 1)
    image_reshape = np.squeeze(image_reshape)
    return(image_reshape)

dest_folder = "/deep_data/datasets/read_curtis/Training-howe"
get_ipython().system(' mkdir $dest_folder')

data_folder = "/deep_data/datasets/read_curtis/Training/"
images_dir = os.path.join(data_folder,'Images')
images_files = [os.path.join(images_dir,f) for f in os.listdir(images_dir) if ".jpg" in f.lower()]

print(os.path.basename(images_files[0]))

for f in images_files:
    img = cv2.imread(f)
    img_how = binarize(img)
    cv2.imwrite(os.path.join(dest_folder, f.lower().partition(".jpg")[0] + "_howe.jpg"), img_how)



img_file = "/deep_data/datasets/ICFHR_Data/general_data/30865/Konzilsprotokolle_B/30865_0011_1063536_r2_r2l1.jpg"
img_c  = cv2.imread(img_file)
plt.imshow(rgb(img_c))

img_how = binarize(img_c)

plt.imshow(img_how, cmap="Greys_r")

# Image 0003 has a lot of background, so it is a good one to test the binarizer
img_file_read = "/deep_data/datasets/read_curtis/Training/Images/Seite0003.JPG"
img_read = cv2.imread(img_file_read)
plt.imshow(rgb(img_read))

img_how_read = binarize(img_read)

plt.imshow(rgb(img_read))

plt.imshow(img_how_read, cmap="Greys_r")

cv2.imwrite("test_howe_read.png", img_how_read)

saved_img_read = cv2.imread("test_howe_read.png", 0)
print(saved_img_read.shape)

saved_img_read[0, 0, 2]

plt.imshow(saved_img_read, cmap="Greys_r")

# The binarizer cuts off the edges a little bit. I can just pad images with whitespace, I think, hopefully it won't make a difference
print(img_c.shape)
print(img_how.shape)
print(img_read.shape)
print(img_how_read.shape)




#img = warp_image(img)

# This looks beautiful with the present parameters of w_mesh 100, h_mesh 40, w_mesh_std 1, h_mesh_std 1. The standard deviation should probably be more to warp the text more.
img_warped = warp_image(img_c, w_mesh_interval=100, h_mesh_interval=40, w_mesh_std=10, h_mesh_std=4)

plt.imshow(rgb(img_warped))

from PIL import Image, ImageOps

b = np.array(Image.open(img_file).convert('L'))

b.shape

plt.imshow(Image.fromarray(warp_image(np.array(Image.open(img_file).convert('L')))))

l = Image.open(img_file).convert('L')

l.shape()

opencvImage = numpy.array(PILImage)

# I say let's just read in the channels, put them in one numpy array, padding edges as necessary. Could even do a for loop to map pixels availabale from howe

# The dimensions should be the same because of the mask!! Plus, I didn't get any errors in making the dataset. A few pixels here or there shouldn't make a difference
img = "/deep_data/datasets/read_curtis/Training/Images/Seite0001.JPG"
img_howe = "/deep_data/datasets/read_curtis/Training-howe/seite0001_howe.jpg"
img_simple = "/deep_data/datasets/read_curtis/Training-imgtxt/seite0001_simplebin.jpg"

image = Image.open(img).convert('L')
howe_img = Image.open(img_howe).convert('L')
bin_img = Image.open(img_simple).convert('L')

img_rgb = Image.open(img).convert("RGB")
print(img_rgb.size)

howe_t_img = Image.new('L', image.size)
howe_t_img.paste(howe_img, howe_img.getbbox())

new_img = Image.merge("RGB", (image, howe_t_img, bin_img))

print(image.size)
print(howe_img.size)
print(bin_img.size)


from PIL import Image

im = Image.open(my_image_file)
a4im = Image.new('RGB',
                 (595, 842),   # A4 at 72dpi
                 (255, 255, 255))  # White
a4im.paste(im, im.getbbox())  # Not centered, top-left corner
a4im.save(outputfile, 'PDF', quality=100)

I, H, B = new_img.split()

plt.imshow(bin_img) #, cmap="Greys_r")

a = np.array(howe_t_img)
b = np.array(H)

b.shape

np.array_equal(a,b)

