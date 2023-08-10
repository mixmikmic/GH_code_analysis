import numpy as np 
import glob
import random 
import math 
from skimage.io import imread 

#for debugging 
import matplotlib.pyplot as plt

import cv2
import os 
for i in range(255):
    arr = np.ones((1080, 1920, 3), dtype=np.float32)*i
 
    num = str(0)*(3-len(str(i))) + str(i) 
    cv2.imwrite(os.path.join('images', num+'.jpg'), arr)

def DataGenerator(img_dir, batch_size=32):
    '''Yields tensor of size [batch_size, 3, 512, 512]. 
    
    Each batch contains [batch_size] number of patches of a single image 
    from the img_dir input directory. Note that patches are reshaped from [512, 512, 3] to 
    [3, 512, 512] which may cause color distortion. 
    
    Args:
    img_dir : string
        directory for images, takes form 'dir/' (default None)
    batch_size : int 
        batch size (default 32)
    '''
    
    #get and verify images are in directory 
    img_dir = glob.glob(img_dir + '*')
    assert len(img_dir) > 0, 'no images found, check directory'
    
    num_batches = math.ceil(len(img_dir)/batch_size) #number of batches   
    start, end = 0, batch_size #frame of images in list img_dir
    batch = np.zeros((batch_size, 3, 512, 512)) #init output batch 
  
    for i in range(num_batches):
        img_batch_files = img_dir[start:end]
        
        for j, img_name in enumerate(img_batch_files): 
            img = imread(img_name)
            
            #random patch 
            x_crop = random.randint(0, (img.shape[0]-512))
            y_crop = random.randint(0, (img.shape[1]-512))
            patch = img[x_crop:x_crop+512, y_crop:y_crop+512, :]
#             plt.imshow(patch); plt.show()
            patch = patch.reshape(1, 3, 512, 512)
            batch[j, :, :, :] = patch 
        
        #clip last batch 
        if i == num_batches - 1:
            batch = batch[:j, :, :, :]
            
        yield batch
        
        #increment images for next iteration 
        start += batch_size
        end += batch_size

#create the generator 
gen = DataGenerator('images/', 32)

#print out the generator output until the StopIteration error
try: 
    for i in range(100): 
        batch = next(gen)
        print ('iteration {}, batch size: {}, batch mean: {}'.format(i, batch.shape, np.mean(batch)))

except StopIteration: 
    print ('reached end of image data')

#create the generator 
gen = DataGenerator('real/', 32)

#print out the generator output until the StopIteration error
try: 
    for i in range(100): 
        batch = next(gen)
        print ('iteration {}, batch size: {}, batch mean: {}'.format(i, batch.shape, np.mean(batch)))

except StopIteration: 
    print ('reached end of image data')

#for GPU support 
import threading

#for image augmentation 
from skimage import transform

'''Augment the image 

1. random angle rotation (+crop out black/resize)
2. random flip about y-axis 
3. random shear 
'''

def rotate(image, angle):
    #random angle between given range
    angle_jitter = (random.random()*angle*2)-angle 
    rotated_image = transform.rotate(image, angle)
    
    #length of pixels lost from rotation
    pix = angle*14
    rotated_image = rotated_image[pix:-pix, :, :]
    rotated_image = transform.resize(rotated_image, image.shape)
    return rotated_image

def shear(image, shear):
    
    # create shear_val from [-shear, shear] 
    shear_val = random.random()*shear*2-shear 
    
    # create Afine transform
    afine = transform.AffineTransform(shear=shear_val)
    
    # apply transform to image data
    sheared_image = transform.warp(image, inverse_map=afine)
    
    #crop out black 
    sheared_image = sheared_image[:, 100:-100, :]
    sheared_image = transform.resize(sheared_image, image.shape)
    return sheared_image

def flip(image):
    flip_prob = random.randint(0, 1) #50/50
    flipped_image = image[::-1] if flip_prob == 1 else image
    return flipped_image

    
    
def DataGenerator(img_dir, batch_size=32, augment=False, angle=12, shear_angle=0.12):
    '''Yields tensor of size [batch_size, 3, 512, 512]. GPU compatible. 
    
    Each batch contains [batch_size] number of patches of a single image 
    from the img_dir input directory. Note that patches are reshaped from [512, 512, 3] to 
    [3, 512, 512] which may cause color distortion.
    
    Additionally it augments the image at each iteration with: random angle rotation,
    random flip about y-axis, random shear. 
    
    Args:
    img_dir : string
        directory for images, takes form 'dir/' (default None)
    batch_size : int 
        batch size (default 32)
    augment : bool 
        if true, augments the image (default False)
    angle : int 
        angle range to rotate about - [-angle, angle] (default 12)
    shear : float 
        shear angle range - [-shear, shear](default 0.12)
    '''
    
    #get and verify images are in directory 
    img_dir = glob.glob(img_dir + '*')
    assert len(img_dir) > 0, 'no images found, check directory'
    
    num_batches = math.ceil(len(img_dir)/batch_size) #number of batches    
    start, end = 0, batch_size #frame of images in list img_dir
    batch = np.zeros((batch_size, 3, 512, 512)) #init output batch 
    
    #lock and release threads at iteration execution 
    with threading.Lock():      
        for i in range(num_batches):
            img_batch_files = img_dir[start:end]

            for j, img_name in enumerate(img_batch_files): 
                img = imread(img_name)
                
                #image augmentation 
                if augment: 
                    img = rotate(img, angle)
                    img = shear(img, shear_angle)
                    img = flip(img)

                #random patch 
                x_crop = random.randint(0, (img.shape[0]-512))
                y_crop = random.randint(0, (img.shape[1]-512))
                patch = img[x_crop:x_crop+512, y_crop:y_crop+512, :]
    #             plt.imshow(patch); plt.show()
                patch = patch.reshape(1, 3, 512, 512)
                batch[j, :, :, :] = patch 
            
            #clip last batch 
            if i == num_batches - 1:
                batch = batch[:j, :, :, :]
                
            yield batch

            #increment images for next iteration 
            start += batch_size
            end += batch_size

#create the generator 
gen = DataGenerator('images/', batch_size=32, augment=True, angle=12, shear_angle=0.12)

#print out the generator output until the StopIteration error
try: 
    for i in range(100): 
        batch = next(gen)
        print ('iteration {}, batch size: {}'.format(i, batch.shape))

except StopIteration: 
    print ('reached end of image data')

