#dependencies
import numpy as np
import utils
import pickle
import os
import re


import tensorflow as tf
import vgg16



######### constants
# required vgg image sizes 
VGG_SIZE_X = 224
VGG_SIZE_Y = 224
VGG_SIZE_Z = 3

# constants for the images
NUM_VIEWS = 40




# to test how the utils.load_image was cutting the image off 
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# parses a foldername to return the relevant info. returns -1 if file is not a png or the filename is ill-formed
def parseFileName(filename):
    
    # ignore DS_Store files 
    if filename == ".DS_Store": 
        return -1
    
    
    fileInfo = filename.split("_")
    
    [target, fileType] = fileInfo[-1].split(".")
    
    # ignore csv's 
    if fileType == "csv": 
        return -1 
    
    if fileType == "png":
        
        subjectID = fileInfo[0] 
        trial = fileInfo[-2]
        
        
        return (subjectID, trial, target) 
        
    print "error: filename syntax incorrect" 
    return -1
        
    

# takes a batch and makes them into manageable 160 rows 
def splitBatches(full_batch, sz):
    num_batches = full_batch.shape[0]/sz; 
    num_remainder = full_batch.shape[0]%sz

    batch = []

    for batch_i in xrange(0,num_batches):
        batch.append(full_batch[batch_i * sz: (batch_i + 1) * sz])


    if num_remainder != 0: 
        batch.append(full_batch[(-1 * num_remainder):])
        
        
    return batch

full_batch = np.empty((0, VGG_SIZE_X, VGG_SIZE_Y, VGG_SIZE_Z), float)
target = [] 
sketch_folder = './sketch_data_small'
for folderName, subfolders, filenames in os.walk(sketch_folder):
    print ('Downloading sketches from: '  + folderName)
    
    # skip the sketch_data folder. 
    if folderName == sketch_folder: 
        continue
    
    

    for filename in filenames: 
        
        
        if parseFileName(filename) != -1: 
            [subjectID_i, trial_i, target_i] = parseFileName(filename)
            target.append(target_i)
            
            img = utils.load_image(folderName + '/' + filename)
            
            # take out the fourth dimension, alpha, which controls transparency
            img = img[:,:,:3]
            
            img = img.reshape(1, VGG_SIZE_X, VGG_SIZE_Y, VGG_SIZE_Z)
            full_batch = np.concatenate((full_batch, img))            
        
#         print ('FILE INSIDE ' + folderName + ':' + filename) 

print full_batch.shape
batch = splitBatches(full_batch, 160);



img1 = utils.load_image("./test_data/limoToSUV_40_15.png.png")

batch1 = img1.reshape((1, 224, 224, 3))


imgplot = plt.imshow(img1)

# to upload multiple images

cars = ['limoToSUV_10','limoToSUV_99','smartToSedan_10','smartToSedan_99'];

batch = np.empty((0, VGG_SIZE_X, VGG_SIZE_Y, VGG_SIZE_Z), float)
for car in cars:  
    for view in xrange(0,NUM_VIEWS):
        imgloc ='https://s3.amazonaws.com/morphrecog-images-1/' + car + '_' + str(view) + '.png.png'
        img = utils.load_image(imgloc)
        img = img.reshape(1, VGG_SIZE_X, VGG_SIZE_Y, VGG_SIZE_Z)
        batch = np.concatenate((batch, img))
        

        
    

# smaller batch for testing first
print batch.shape[0]
batch_mini = batch[:4,:,:,:]
print batch_mini.shape[0]


# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
with tf.device('/cpu:0'):
#with tf.device('/gpu:0'): # to add this when I use the gpu version 
    with tf.Session() as sess:
        image = tf.placeholder("float", [batch_mini.shape[0], VGG_SIZE_X, VGG_SIZE_Y, VGG_SIZE_Z])
        
        feed_dict = {image: batch_mini}

        vgg = vgg16.Vgg16()
        with tf.name_scope("content_vgg"):
            vgg.build(image)

        act_wanted = [vgg.pool1, vgg.pool2, vgg.prob]
        act = sess.run(act_wanted, feed_dict=feed_dict)

        for i in xrange(0, batch_mini.shape[0]):
            utils.print_prob(act[2][i], './synset.txt')
        
       

    

act_test = pickle.load(open('act_test.p', 'rb'))



