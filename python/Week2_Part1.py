get_ipython().magic('matplotlib inline')

#import typical packages I'll be using
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10  #boiler plate to set the size of the figures
plt.gray()  #start with gray scale image

#Load a test image - convert to a single channel, and display as a gray scale image
#im = cv2.imread("camera_man.png")[:,:,0]
#Only use part of the image to start
#im = im[32:96,64:128]

#Or use lena - useful had less compression artifacts than my camera_man
im = cv2.imread("lena.tiff")
#convert for matplotlib from brg to Y CR CB for display
im = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)
#use only the Y channel
im = im[:,:,0]
#Only use part of the image to start
im = im[232:296,232:296]

plt.imshow(im)

w,h = im.shape
block_size = 8

blocks = np.zeros((block_size,block_size,w/block_size,h/block_size),np.int)
for r in range(h/block_size):
    for c in range(w/block_size):
        blocks[r,c] = (im[r*block_size : (r+1)*block_size, c*block_size : (c+1)*block_size])

f, axarr = plt.subplots(h/block_size, w/block_size)
f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.01, hspace=.01)

for r in range(h/block_size):
    for c in range(w/block_size):
        axarr[r,c].imshow(blocks[r,c],vmin=0, vmax=255)
        axarr[r,c].axis('equal')
        axarr[r,c].axis('off')

dct = np.empty_like(blocks).astype(np.float32)

for r in range(h/block_size):
    for c in range(w/block_size):
        dct[r,c] = cv2.dct(np.float32(blocks[r,c]))  #opencv wants 32 bit floats for cvt

# reviewing the effects of the dct - here is a block
blocks[4,0]

# here is the computed dct for the same 8x8 block
dct[0,0]

#quantize matrix from book 8.30b

normalization = np.asarray(
[16, 11, 10, 16, 24, 40, 51, 61, 
 12, 12, 14, 19, 26, 58, 60, 55, 
 14, 13, 16, 24, 40, 57, 69, 56, 
 14, 17, 22, 29, 51, 87, 80, 62,
 18, 22, 37, 56, 68, 109, 103, 77,
 24, 35, 55, 64, 81, 104, 113, 92,
 49, 64, 78, 87, 103, 121, 120, 101,
 72, 92, 95, 98, 112, 100, 103, 99]
).reshape(8,8)

quantized = np.empty_like(dct)

for r in range(h/block_size):
    for c in range(w/block_size):
        quantized[r,c] = dct[r,c]/normalization

quantized = quantized.astype(np.int)

quantized[0,0]

inverted = np.empty_like(quantized)

for r in range(h/block_size):
    for c in range(w/block_size):
        inverted[r,c] = cv2.idct(np.float32(quantized[r,c]*normalization))  #invert quantization and dct

f, axarr = plt.subplots(h/block_size, w/block_size)
f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.01, hspace=.01)

for r in range(h/block_size):
    for c in range(w/block_size):
        axarr[r,c].imshow(inverted[r,c],vmin=0, vmax=255)
        axarr[r,c].axis('equal')
        axarr[r,c].axis('off')

#Left side pre-compression, right side post-compression
f, axarr = plt.subplots(4,2)
for i in range(4):
        axarr[i,0].imshow(blocks[2+i,4],vmin=0, vmax=255)
        axarr[i,0].axis('equal')
        axarr[i,0].axis('off')
        
        axarr[i,1].imshow(inverted[2+i,4],vmin=0, vmax=255)
        axarr[i,1].axis('equal')
        axarr[i,1].axis('off')

#An example showing the movement of values from image -> dct -> quantized -> invert_quantiziation -> invert_dct 
print blocks[4,7,0]
print dct[4,7,0].astype(np.int)
print quantized[4,7,0]
print (quantized[4,7]*normalization)[0]
print inverted[4,7,0]

fft = np.empty_like(blocks).astype(np.complex)

for r in range(h/block_size):
    for c in range(w/block_size):
        fft[r,c] = np.fft.fft(blocks[r,c])
        #cv2.dft(np.float32(blocks[r,c]))  #opencv wants 32 bit floats for dft
        
quantized = np.empty_like(fft)

for r in range(h/block_size):
    for c in range(w/block_size):
        quantized[r,c] = fft[r,c]/normalization

inverted = np.empty_like(quantized).astype(np.complex)

for r in range(h/block_size):
    for c in range(w/block_size):
        inverted[r,c] = np.fft.ifft(quantized[r,c]*normalization)
        #cv2.idft(np.float32(quantized[r,c]*normalization))  #invert quantization and dft
    
inverted = inverted.astype(np.int)

f, axarr = plt.subplots(h/block_size, w/block_size)
f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.01, hspace=.01)

for r in range(h/block_size):
    for c in range(w/block_size):
        axarr[r,c].imshow(inverted[r,c],vmin=0, vmax=255)
        axarr[r,c].axis('equal')
        axarr[r,c].axis('off')

#An example showing the movement of values from image -> dct -> quantized -> invert_quantiziation -> invert_dct 
print blocks[6,7,0]
print fft[6,7,0]
print quantized[6,7,0]
print (quantized[6,7]*normalization)[0]
print inverted[6,7,0]

quantized = np.empty_like(blocks)

for r in range(h/block_size):
    for c in range(w/block_size):
        quantized[r,c] = blocks[r,c]/normalization

quantized = quantized.astype(np.int)

inverted = np.empty_like(quantized)

for r in range(h/block_size):
    for c in range(w/block_size):
        inverted[r,c] = (quantized[r,c]*normalization)
    
inverted = inverted.astype(np.int)

f, axarr = plt.subplots(h/block_size, w/block_size)
f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.01, hspace=.01)

for r in range(h/block_size):
    for c in range(w/block_size):
        axarr[r,c].imshow(inverted[r,c],vmin=0, vmax=255)
        axarr[r,c].axis('equal')
        axarr[r,c].axis('off')

