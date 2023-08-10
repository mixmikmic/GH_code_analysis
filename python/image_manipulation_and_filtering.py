# Import the required modules
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
from PIL import Image
from skimage import data
import scipy
# from IPython.html.widgets import interact, fixed
from ipywidgets import interact, fixed, FloatSlider, IntSlider,FloatRangeSlider, Label

# Load an image
rgb_image_int = Image.open("color_circle.png")
rgb_image_float= np.asarray(rgb_image_int,dtype=float)/255.0 # Convert the image to numpy array
print(rgb_image_float.shape)
rgb_image_float = rgb_image_float[:,:,:3] # Make sure image has three layers
print(rgb_image_float.shape)
plt.imshow(rgb_image_float)
plt.show()

def demo_rgb_to_hsv(original_image,reduce_intensity_factor=0.5):
    original_rgb_image_float= np.asarray(original_image,dtype=float)/255.0 # Convert the image float
    original_rgb_image_float = original_rgb_image_float[:,:,:3] # Make sure image has three layers
    hsv_image=matplotlib.colors.rgb_to_hsv(original_rgb_image_float)
    hsv_image_processed=hsv_image.copy()
    hsv_image_processed[:,: ,2]=hsv_image[:,: ,2]*reduce_intensity_factor
    rgb_image_processed=matplotlib.colors.hsv_to_rgb(hsv_image_processed)
    fig1, axes_array = plt.subplots(1, 2)
    fig1.set_size_inches(8,4)
    image_plot = axes_array[0].imshow(original_rgb_image_float) # Show the RGB image
    axes_array[0].axis('off')
    axes_array[0].set(title='RGB Image')
    image_plot = axes_array[1].imshow(rgb_image_processed) # Show the gray image
    axes_array[1].axis('off')
    axes_array[1].set(title='Reduced Intensity Image')
    plt.show()

rgb_image_int = Image.open("color_circle.png")
interact(demo_rgb_to_hsv,original_image=fixed(rgb_image_int),
         reduce_intensity_factor=FloatSlider(min=0., max=1., step=0.05,value=0.5,description='Reduce Intensity'));

def demo_image_diaply_range(original_image,range=[0.3,.7]):
    rgb_image_float= np.asarray(original_image,dtype=float)/255.0 # Convert the image to numpy array
    gray_image_int=original_image.convert('L')
    gray_image_float= np.asarray(gray_image_int,dtype=float)/255.0 # Normalize the image to be between 0 to 1
    fig1, axes_array = plt.subplots(1, 3)
    fig1.set_size_inches(9,3)
    image_plot = axes_array[0].imshow(rgb_image_float) # Show the RGB image
    axes_array[0].axis('off')
    axes_array[0].set(title='RGB image')
    image_plot = axes_array[1].imshow(gray_image_float,cmap=plt.cm.gray) # Show the gray image
    axes_array[1].axis('off')
    axes_array[1].set(title='Gray image')
    image_plot = axes_array[2].imshow(gray_image_float,cmap=plt.cm.gray,vmin=range[0], vmax=range[1]) # Limit the range
    axes_array[2].axis('off')
    axes_array[2].set(title='Range limited gray image')
    plt.show()
    
rgb_image_int = Image.open("color_circle.png")
interact(demo_image_diaply_range,original_image=fixed(rgb_image_int),
         range=FloatRangeSlider(min=0., max=1., step=0.05, value=[0.3, 0.7]),
        continuous_update=False, description='Range');

kernel = np.array([[ 1.,  2,  1],[ 0,  0,  0],[-1,-2,-1]])
kernel_sum=np.sum(kernel)
kernel= kernel/kernel_sum if kernel_sum else kernel
plt.rcParams['image.interpolation'] = 'none'
plt.imshow(kernel, cmap=plt.cm.gray);
plt.axis('off')
for row in range(np.shape(kernel)[0]):
    for col in range(np.shape(kernel)[1]):
        plt.text(col,row,'{:0.2f}'.format(kernel[row][col]),ha='center',color='red')
plt.show()

from scipy.ndimage import convolve
plt.rcParams['image.interpolation'] = 'none'
def convolve_two_images_and_display_the_results(kernel,original_image):
    # Normalize the kernel
    kernel_sum=abs(np.sum(kernel))
    kernel= kernel/kernel_sum if kernel_sum else kernel
    filtered_image = scipy.ndimage.convolve(original_image, kernel)
    fig1, axes_array = plt.subplots(1, 3)
    fig1.set_size_inches(9,3)
    image_plot = axes_array[0].imshow(original_image ,cmap=plt.cm.gray) # Show the original image
    axes_array[0].axis('off')
    axes_array[0].set(title='Original image')
    image_plot = axes_array[1].imshow(kernel,cmap=plt.cm.gray) # Show the kernel
    for row in range(np.shape(kernel)[0]):
        for col in range(np.shape(kernel)[1]):
            axes_array[1].text(col,row,'{:0.2f}'.format(kernel[row][col]),ha='center',color='red')
    axes_array[1].axis('off')
    axes_array[1].set(title='Kernel image')
    image_plot = axes_array[2].imshow(filtered_image,cmap=plt.cm.gray) # Limit the range
    axes_array[2].axis('off')
    axes_array[2].set(title='Filtered Image')
    plt.show()
    
kernel = np.array([[ 1.,  2,  1],[ 0,  0,  0],[-1,-2,-1]])
original_image = data.camera()/255.
convolve_two_images_and_display_the_results(kernel,original_image)

def demo_kernel_size(original_image,kernel_size=3):
    kernel = np.ones((kernel_size,kernel_size))
    convolve_two_images_and_display_the_results(kernel,original_image)
current_image = data.camera()/255.
interact(demo_kernel_size,original_image=fixed(current_image),
         kernel_size=IntSlider(min=1, max=51., step=2,value=3));

def display_1d_gaussian(mean=0.0,sigma=0.5):
    x=np.linspace(-10,10,1000)
    y= (1/np.sqrt(2*np.pi*sigma**2))*np.exp(-((x-mean)**2)/(2*sigma**2))
    fig, axes1 = plt.subplots(1, 1)
    fig.set_size_inches(6,3)
    axes1.set(xlabel="X",ylabel="Y",title='Gaussian Curve',ylim=(0,1))
    plt.grid(True)
    axes1.plot(x,y,color='gray')
    plt.fill_between(x,y,0,color='#c0f0c0')
    plt.show()
interact(display_1d_gaussian,mean=FloatSlider(min=-10., max=10., step=0.1),
        sigma=FloatSlider(min=0.1, max=10, step=0.1, value=0.5));

import scipy.stats
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
def display_gaussian_kernel(sigma=1.0):
    X = np.linspace(-5, 5, 400)
    Y = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)
    mu = np.array([0.0, 0.0])
    covariance = np.diag(np.array([sigma, sigma])**2)
    XY = np.column_stack([X.flat, Y.flat])
    z = scipy.stats.multivariate_normal.pdf(XY, mean=mu, cov=covariance)
    Z = z.reshape(X.shape)

    # Plot the surface.
    fig = plt.figure()
    fig.set_size_inches(8,4)
    ax1 = fig.add_subplot(121)
    ax1.imshow(Z)
    ax2 = fig.add_subplot(122, projection='3d')

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # axes_array[0].set(projection='3d')
    surf = ax2.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Customize the z axis.
    ax2.set_zlim(0, .2)
    ax2.zaxis.set_major_locator(LinearLocator(10))
    ax2.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
interact(display_gaussian_kernel,sigma=FloatSlider(min=1, max=3, step=0.05,value=1.0,continuous_update=False));

def gaussian_filter_and_display_results(original_image, sigma):
    filtered_image=scipy.ndimage.filters.gaussian_filter(original_image, 
            sigma=sigma, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
    fig1, axes_array = plt.subplots(1, 2)
    fig1.set_size_inches(8,4)
    image_plot = axes_array[0].imshow(original_image,cmap=plt.cm.gray) 
    axes_array[0].axis('off')
    axes_array[0].set(title='Original Image')
    image_plot = axes_array[1].imshow(filtered_image,cmap=plt.cm.gray)
    axes_array[1].axis('off')
    axes_array[1].set(title='Filtered Image')
    plt.show()

current_image = data.camera()/255.
interact(gaussian_filter_and_display_results,original_image=fixed(current_image),
         sigma=FloatSlider(min=0.0, max=10, step=0.1,continuous_update=False));

original_image = data.camera()/255.
filtered_image=scipy.ndimage.filters.gaussian_filter(original_image, 
        sigma=15, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
sharpened_image=original_image-filtered_image
fig1, axes_array = plt.subplots(1, 3)
fig1.set_size_inches(9,3)
image_plot = axes_array[0].imshow(original_image ,cmap=plt.cm.gray) # Show the original image
axes_array[0].axis('off')
axes_array[0].set(title='Original image')
image_plot = axes_array[1].imshow(filtered_image,cmap=plt.cm.gray) # Show the filtered image
axes_array[1].axis('off')
axes_array[1].set(title='Blurred image')
image_plot = axes_array[2].imshow(sharpened_image,cmap=plt.cm.gray) # Show the sharpened image
axes_array[2].axis('off')
axes_array[2].set(title='Filtered Image')
plt.show()

def median_filter_demo(original_image,filter_size=5,noise_percent=10):
    noise=np.random.rand(*original_image.shape)
    noisy_image=original_image.copy()
    noisy_image[noise>(1-noise_percent*.01)]=1.0
    median_filtered_image=scipy.ndimage.filters.median_filter(noisy_image, 
                    size=filter_size, footprint=None, output=None, mode='reflect', cval=0.0, origin=0)
    fig1, axes_array = plt.subplots(1, 3)
    fig1.set_size_inches(9,3)
    image_plot = axes_array[0].imshow(original_image ,cmap=plt.cm.gray) # Show the original image
    axes_array[0].axis('off')
    axes_array[0].set(title='Original image')
    image_plot = axes_array[1].imshow(noisy_image,cmap=plt.cm.gray) # Show the filtered image
    axes_array[1].axis('off')
    axes_array[1].set(title='Noisy image')
    image_plot = axes_array[2].imshow(median_filtered_image,cmap=plt.cm.gray) # Show the sharpened image
    axes_array[2].axis('off')
    axes_array[2].set(title='Filtered Image')
    plt.show()
    return
current_image = data.camera()/255.
# current_image=current_image[200:220,200:220]
interact(median_filter_demo,original_image=fixed(current_image),
         filter_size=IntSlider(min=3, max=15, step=2,continuous_update=False,description='Filter Size'),
         noise_percent=IntSlider(min=0, max=100, step=1,value=10,continuous_update=False,description='Noise %'));

def binarize_image_and_display_results(input_image, threshold):
#     print("Threshold=",threshold)
    binarized_image=np.zeros(input_image.shape)
    binarized_image[input_image>=threshold]=1.0
    fig1, axes_array = plt.subplots(1, 2)
    fig1.set_size_inches(8,4)
    image_plot = axes_array[0].imshow(input_image,cmap=plt.cm.gray) 
    axes_array[0].axis('off')
    axes_array[0].set(title='Original Image')
    image_plot = axes_array[1].imshow(binarized_image,cmap=plt.cm.gray)
    axes_array[1].axis('off')
    axes_array[1].set(title='Binarized Image')
    plt.show()
original_image = data.camera()/255.
interact(binarize_image_and_display_results,input_image=fixed(original_image),
         threshold=FloatSlider(min=0.1, max=1.0, step=0.05,value=.5,continuous_update=False,description='Threshold'));

from skimage import data
def erosion_demo(binarized_image,kernel_size):
    eroded_image=scipy.ndimage.morphology.binary_erosion(binarized_image, 
                    structure=np.ones((kernel_size,kernel_size)), iterations=1, mask=None,
                    border_value=0, origin=0, brute_force=False)
    fig1, axes_array = plt.subplots(1, 3)
    fig1.set_size_inches(9,3)
    image_plot = axes_array[0].imshow(original_image ,cmap=plt.cm.gray) # Show the original image
    axes_array[0].axis('off')
    axes_array[0].set(title='Original image')
    image_plot = axes_array[1].imshow(binarized_image,cmap=plt.cm.gray) 
    axes_array[1].axis('off')
    axes_array[1].set(title='Binarized image')
    image_plot = axes_array[2].imshow(eroded_image,cmap=plt.cm.gray) 
    axes_array[2].axis('off')
    axes_array[2].set(title='Eroded Image')
    plt.show()
original_image=np.asarray(data.coins())/255.
current_binarized_image=np.where(original_image>0.5,1.,0.)


interact(erosion_demo,binarized_image=fixed(current_binarized_image),
         kernel_size=IntSlider(min=3, max=15, step=2,value=3,
            description='Kernel Size:',slider_color='red',continuous_update=False));

from skimage import data
def dilation_demo(binarized_image,kernel_size):
    eroded_image=scipy.ndimage.morphology.binary_dilation(binarized_image, 
                    structure=np.ones((kernel_size,kernel_size)), iterations=1, mask=None,
                    border_value=0, origin=0, brute_force=False)
    fig1, axes_array = plt.subplots(1, 3)
    fig1.set_size_inches(9,3)
    image_plot = axes_array[0].imshow(original_image ,cmap=plt.cm.gray) # Show the original image
    axes_array[0].axis('off')
    axes_array[0].set(title='Original image')
    image_plot = axes_array[1].imshow(binarized_image,cmap=plt.cm.gray) 
    axes_array[1].axis('off')
    axes_array[1].set(title='Binarized image')
    image_plot = axes_array[2].imshow(eroded_image,cmap=plt.cm.gray) 
    axes_array[2].axis('off')
    axes_array[2].set(title='Dilated Image')
    plt.show()
original_image=np.asarray(data.coins())/255.
current_binarized_image=np.where(original_image>0.5,1.,0.)


interact(dilation_demo,binarized_image=fixed(current_binarized_image),
         kernel_size=IntSlider(min=3, max=15, step=2,value=3,
            description='Kernel Size:',slider_color='red',continuous_update=False));

from skimage import data
def close_and_open_demo(binarized_image,kernel_size):
    closed_image=scipy.ndimage.morphology.binary_closing(current_binarized_image, 
             structure=np.ones((kernel_size,kernel_size)), iterations=1, output=None, origin=0)
    opened_image=scipy.ndimage.morphology.binary_opening(current_binarized_image, 
                 structure=np.ones((kernel_size,kernel_size)), iterations=1, output=None, origin=0)
    fig1, axes_array = plt.subplots(1, 3)
    fig1.set_size_inches(9,3)
    image_plot = axes_array[0].imshow(current_binarized_image ,cmap=plt.cm.gray) 
    axes_array[0].axis('off')
    axes_array[0].set(title='Binarized Image')
    image_plot = axes_array[1].imshow(closed_image,cmap=plt.cm.gray) 
    axes_array[1].axis('off')
    axes_array[1].set(title='Closed Image')
    image_plot = axes_array[2].imshow(opened_image,cmap=plt.cm.gray) 
    axes_array[2].axis('off')
    axes_array[2].set(title='Opened Image')
    plt.show()

original_image=np.asarray(data.coins())/255.
current_binarized_image=np.where(original_image>0.5,1.,0.)
interact(close_and_open_demo,binarized_image=fixed(current_binarized_image),
         kernel_size=IntSlider(min=3, max=15, step=2,value=3,
            description='Kernel Size:',slider_color='red',continuous_update=False));

original_image = data.camera()/255.
flipud_image=np.flipud(original_image)
fliplr_image=np.fliplr(original_image)
rotated_image=scipy.ndimage.rotate(original_image,45)
resized_image=scipy.misc.imresize(original_image, 0.5, interp='bilinear', mode=None)
rows,cols=original_image.shape
croped_image = original_image[int(rows / 3): -int(rows / 3), int(cols / 4): - int(cols / 4)]
fig1, axes_array = plt.subplots(2, 3)
fig1.set_size_inches(9,6)
image_plot = axes_array[0][0].imshow(original_image ,cmap=plt.cm.gray) 
axes_array[0][0].set(title='Original')
image_plot = axes_array[0][1].imshow(flipud_image,cmap=plt.cm.gray) 
axes_array[0][1].set(title='Flipped up-down')
image_plot = axes_array[0][2].imshow(fliplr_image,cmap=plt.cm.gray) 
axes_array[0][2].set(title='Flipped left-right')
image_plot = axes_array[1][0].imshow(rotated_image,cmap=plt.cm.gray) 
axes_array[1][0].set(title='Rotated')
image_plot = axes_array[1][1].imshow(resized_image,cmap=plt.cm.gray) 
axes_array[1][1].set(title='Resized')
image_plot = axes_array[1][2].imshow(croped_image,cmap=plt.cm.gray) 
axes_array[1][2].set(title='Cropped')
plt.show()

