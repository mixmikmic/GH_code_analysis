# Load necessary packages
import numpy as np
import numpy.fft as fft

import skimage.measure as sm

import matplotlib.pyplot as plt
import scipy.misc as misc
from math import pi

#Read in source image
source = plt.imread("./Albert_Einstein_Head.jpg",format="jpg")

# Because the original image is too large downsample the image by a factor of 20
source = sm.block_reduce(source, (20,20), np.mean)

#Show this figure
fig = plt.figure(figsize=(5,5))
plt.imshow(source, cmap = "gray")
plt.colorbar()
plt.title("Image of Albert Einstein")
print("The image size is %d by %d"%(source.shape[0], source.shape[1]))

#Pad image to simulate oversampling

## pad_len is the size of the support
pad_len = len(source)
padded = np.pad(source, ((pad_len, pad_len),(pad_len, pad_len)), 'constant', 
                constant_values=((0,0),(0,0)))

#Show this figure
fig = plt.figure(figsize=(5,5))
plt.imshow(padded, cmap="gray")
plt.colorbar()
plt.title("Image of Albert Einstein")
print("The image size is %d by %d"%(padded.shape[0], padded.shape[1]))

# Calculate the fourier transformation
ft = fft.fft2(padded)

#simulate diffraction pattern
magnitude = np.abs(ft)

length, width = padded.shape

#keep track of where the image is vs the padding
mask = np.ones((pad_len+2,pad_len+2))
mask = np.pad(mask, ((pad_len-1, pad_len-1),(pad_len-1, pad_len-1)), 'constant', 
                constant_values=((0,0),(0,0)))

#Show this mask
fig = plt.figure(figsize=(5,5))
plt.imshow(mask, cmap="gray")
plt.colorbar()
plt.title("Image of Albert Einstein")
print("The image size is %d by %d"%(mask.shape[0], mask.shape[1]))

#Initial guess using random phase info
guess = diffract * np.exp(1j * np.random.rand(length,width) * 2 * pi)

#number of iterations
r = 801

#step size parameter
beta = 0.8

#Finishes the first round because it's different from the other round
prev = None

# Begin the loop
for s in range(0,r):
    #apply fourier domain constraints
    update = diffract * np.exp(1j * np.angle(guess)) 
    
    inv = fft.ifft2(update)
    inv = np.real(inv)
    if prev is None:
        prev = inv
        
    #apply real-space constraints
    temp = inv
    
    ## This part need to be vectorize
    for i in range(0,length):
        for j in range(0,width):
            #image region must be positive
            if inv[i,j] < 0 and mask[i,j] == 1:
                inv[i,j] = prev[i,j] - beta*inv[i,j]
            #push support region intensity toward zero
            if mask[i,j] == 0:
                inv[i,j] = prev[i,j] - beta*inv[i,j]
    
    
    prev = temp
    
    guess = fft.fft2(inv)
        
    #save an image of the progress
    if s % 10 == 0:
        misc.imsave("./save/progress" + str(s) +
                    ".png", prev)
        print(s)



