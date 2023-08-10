get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
import theano
import theano.tensor as T
import numpy as np
import SRCNN_Theano
import os
import preprocessing
from scipy import ndimage,misc

#Create Validation_Subsamples_RGB_4 and Validation_Subsamples_RGB_4_GT folders
preprocessing.create_subimages('/home/jon/Documents/Neural_Networks/project/SRCNN_Theano/Data/Set5',
'/home/jon/Documents/Neural_Networks/project/SRCNN_Theano/Data/Validation_Subsamples_RGB_4',
output_type ='RGB',upsampling_factor = 4)
preprocessing.create_subimages('/home/jon/Documents/Neural_Networks/project/SRCNN_Theano/Data/Set14',
'/home/jon/Documents/Neural_Networks/project/SRCNN_Theano/Data/Test_Subsamples_RGB_4',
output_type ='RGB',upsampling_factor = 4)
preprocessing.create_subimages('/home/jon/Documents/Neural_Networks/project/SRCNN_Theano/Data/Training_Full',
'/home/jon/Documents/Neural_Networks/project/SRCNN_Theano/Data/Training_Subsamples_RGB_4',
output_type ='RGB',upsampling_factor = 4)

import SRCNN_Theano

#load dataset

#load training
data_x = SRCNN_Theano.load_dataset('Data/Training_Subsamples_RGB_4','data_x')
data_y = SRCNN_Theano.load_dataset('Data/Training_Subsamples_RGB_4_gt','data_y')
valid_x = SRCNN_Theano.load_dataset('Data/Validation_Subsamples_RGB_4','data_x')
valid_y = SRCNN_Theano.load_dataset('Data/Validation_Subsamples_RGB_4_gt','data_y')
test_x = SRCNN_Theano.load_dataset('Data/Test_Subsamples_RGB_4','data_x')
test_y = SRCNN_Theano.load_dataset('Data/Test_Subsamples_RGB_4_gt','data_y')
print "done loading\n\n"
print "data_x: " + str(data_x.shape)
print "data_y: " + str(data_y.shape)
print "valid_x: " + str(valid_x.shape)
print "valid_y: " + str(valid_y.shape)
print "test_x: " + str(test_x.shape)
print "test_y: " + str(test_y.shape)

#Bicubic interp to save computation during training
upsampled_x = preprocessing.upsample(data_x) #33,33,3 input images expected
up_val_x = preprocessing.upsample(valid_x) #33,33,3 input images expected
up_test_x = preprocessing.upsample(test_x) #33,33,3 input images expected

#Reshape for training,valid,test
upsampled_x = upsampled_x.reshape((22092,33*33*3))
data_y = data_y.reshape((22092,33*33*3))
up_val_x = up_val_x.reshape((2488,33*33*3))
valid_y = valid_y.reshape((2488,33*33*3))
up_test_x = up_test_x.reshape((14851,33*33*3))
test_y = test_y.reshape((14851,33*33*3))

print upsampled_x.shape
print data_y.shape
print up_val_x.shape
print up_test_x.shape

shared_x = theano.shared(np.asarray(upsampled_x,
                                       dtype=theano.config.floatX),
                         borrow=True)
shared_y = theano.shared(np.asarray(data_y,
                                       dtype=theano.config.floatX),
                         borrow=True)
shared_val_x = theano.shared(np.asarray(up_val_x,
                                       dtype=theano.config.floatX),
                         borrow=True)
shared_val_y = theano.shared(np.asarray(valid_y,
                                       dtype=theano.config.floatX),
                         borrow=True)
shared_test_x = theano.shared(np.asarray(up_test_x,
                                       dtype=theano.config.floatX),
                         borrow=True)
shared_test_y = theano.shared(np.asarray(test_y,
                                       dtype=theano.config.floatX),
                         borrow=True)

import SRCNN_Theano
batch_size = 20
n_epochs = 10

lr = 3e-5
l2s = [0,.1,1e-2,1e-3,1e-4]
for l2 in l2s:
    print "\n\n ****************************** l2 = " + str(l2) +"******************************************"
    learning_rate = lr

    n_train_batches = upsampled_x.shape[0]/batch_size
    n_valid_batches = up_val_x.shape[0]/batch_size
    n_test_batches = up_test_x.shape[0]/batch_size



    train_model,val_model,test_model = SRCNN_Theano.train_BN_SRCNN(shared_x,shared_y,
                             shared_val_x,shared_val_y,
                             shared_test_x,shared_test_y,
                            n_train_batches, n_valid_batches, n_test_batches, 
                             n_epochs, batch_size,learning_rate,l2,upsampling_factor=4)

batch_size = 20
n_epochs = 10

lrs = [1e-2,1e-3,1e-4,3e-5]
l2 = 0
for lr in lrs:
    print "\n\n ****************************** lr = " + str(lr) +"******************************************"
    learning_rate = lr

    n_train_batches = upsampled_x.shape[0]/batch_size
    n_valid_batches = up_val_x.shape[0]/batch_size
    n_test_batches = up_test_x.shape[0]/batch_size



    train_model,val_model,test_model = SRCNN_Theano.train_BN_SRCNN(shared_x,shared_y,
                             shared_val_x,shared_val_y,
                             shared_test_x,shared_test_y,
                            n_train_batches, n_valid_batches, n_test_batches, 
                             n_epochs, batch_size,learning_rate,l2,upsampling_factor=4)

import SRCNN_Theano
batch_size = 20
n_epochs = 10

lr = 3e-5
l2s = [0,.1,1e-2,1e-3,1e-4]
for l2 in l2s:
    print "\n\n ****************************** l2 = " + str(l2) +"******************************************"
    learning_rate = lr

    n_train_batches = upsampled_x.shape[0]/batch_size
    n_valid_batches = up_val_x.shape[0]/batch_size
    n_test_batches = up_test_x.shape[0]/batch_size



    train_model,val_model,test_model = SRCNN_Theano.train_SRCNN(shared_x,shared_y,
                             shared_val_x,shared_val_y,
                             shared_test_x,shared_test_y,
                            n_train_batches, n_valid_batches, n_test_batches, 
                             n_epochs, batch_size,learning_rate,l2,upsampling_factor=4)





batch_size = 20
n_epochs = 10
lrs = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7]
for lr in lrs:
    print "\n\n ****************************** lr = " + str(lr) +"******************************************"
    learning_rate = lr

    n_train_batches = upsampled_x.shape[0]/batch_size
    n_valid_batches = up_val_x.shape[0]/batch_size
    n_test_batches = up_test_x.shape[0]/batch_size



    train_model,val_model,test_model = SRCNN_Theano.train_SRCNN(shared_x,shared_y,
                             shared_val_x,shared_val_y,
                             shared_test_x,shared_test_y,
                            n_train_batches, n_valid_batches, n_test_batches, 
                             n_epochs, batch_size,learning_rate,upsampling_factor=4)

batch_size = 20
n_epochs = 50
lrs = [1e-4]
for lr in lrs:
    print "\n\n ****************************** lr = " + str(lr) +"******************************************"
    learning_rate = lr

    n_train_batches = upsampled_x.shape[0]/batch_size
    n_valid_batches = up_val_x.shape[0]/batch_size
    n_test_batches = up_test_x.shape[0]/batch_size



    train_model,val_model,test_model = SRCNN_Theano.train_SRCNN(shared_x,shared_y,
                             shared_val_x,shared_val_y,
                             shared_test_x,shared_test_y,
                            n_train_batches, n_valid_batches, n_test_batches, 
                             n_epochs, batch_size,learning_rate,upsampling_factor=4)

###rms with xavier

batch_size = 20
n_epochs = 10
lrs = [.1,.07,.05,.03,.007,.005,.002,.0009,.0007,.0004,.0001]
for lr in lrs:
    print "\n\n ****************************** lr = " + str(lr) +"******************************************"
    learning_rate = lr

    n_train_batches = upsampled_x.shape[0]/batch_size
    n_valid_batches = up_val_x.shape[0]/batch_size
    n_test_batches = up_test_x.shape[0]/batch_size



    train_model,val_model,test_model = SRCNN_Theano.train_SRCNN(shared_x,shared_y,
                             shared_val_x,shared_val_y,
                             shared_test_x,shared_test_y,
                            n_train_batches, n_valid_batches, n_test_batches, 
                             n_epochs, batch_size,learning_rate,upsampling_factor=4)

#RMS with xavier

batch_size = 20
n_epochs = 10
lrs = [1e-4,8e-5,5e-5,3e-5,1e-5,1e-6,1e-7]
for lr in lrs:
    print "\n\n ****************************** lr = " + str(lr) +"******************************************"
    learning_rate = lr

    n_train_batches = upsampled_x.shape[0]/batch_size
    n_valid_batches = up_val_x.shape[0]/batch_size
    n_test_batches = up_test_x.shape[0]/batch_size



    train_model,val_model,test_model = SRCNN_Theano.train_SRCNN(shared_x,shared_y,
                             shared_val_x,shared_val_y,
                             shared_test_x,shared_test_y,
                            n_train_batches, n_valid_batches, n_test_batches, 
                             n_epochs, batch_size,learning_rate,upsampling_factor=4)

#RMS with xavier

batch_size = 20
n_epochs = 100
lrs = [3e-5]
for lr in lrs:
    print "\n\n ****************************** lr = " + str(lr) +"******************************************"
    learning_rate = lr

    n_train_batches = upsampled_x.shape[0]/batch_size
    n_valid_batches = up_val_x.shape[0]/batch_size
    n_test_batches = up_test_x.shape[0]/batch_size

    train_model,val_model,test_model = SRCNN_Theano.train_SRCNN(shared_x,shared_y,
                             shared_val_x,shared_val_y,
                             shared_test_x,shared_test_y,
                            n_train_batches, n_valid_batches, n_test_batches, 
                             n_epochs, batch_size,learning_rate,upsampling_factor=4)

#RMS with xavier
batch_size = 20
n_epochs = 100
lrs = [3e-5]
for lr in lrs:
    print "\n\n ****************************** lr = " + str(lr) +"******************************************"
    learning_rate = lr

    n_train_batches = upsampled_x.shape[0]/batch_size
    n_valid_batches = up_val_x.shape[0]/batch_size
    n_test_batches = up_test_x.shape[0]/batch_size

    train_model,val_model,test_model = SRCNN_Theano.train_SRCNN(shared_x,shared_y,
                             shared_val_x,shared_val_y,
                             shared_test_x,shared_test_y,
                            n_train_batches, n_valid_batches, n_test_batches, 
                             n_epochs, batch_size,learning_rate,1e-4,upsampling_factor=4)

reconstructed_imgs = np.zeros((14851, 3, 21, 21))
for i in xrange(n_test_batches):
     cost,MSE_per_pixel,psnr,reconstucted_patches = test_model(i)
     reconstructed_imgs[i*batch_size:(i+1)*batch_size,:,:,:] = reconstucted_patches

SRCNN_Theano.rebuild_images(reconstructed_imgs,'/home/jon/Documents/Neural_Networks/project/SRCNN_Theano/Data/Test_Subsamples_RGB_4',patch_dim=21,dataset='test')
        

up_test_x = preprocessing.upsample(test_x) #33,33,3 input images expected
SRCNN_Theano.rebuild_images(up_test_x,'/home/jon/Documents/Neural_Networks/project/SRCNN_Theano/Data/Test_Subsamples_RGB_4',patch_dim=33,dataset='test')



#SGD with xavier

batch_size = 20
n_epochs = 50
lrs = [1e-4]
for lr in lrs:
    print "\n\n ****************************** lr = " + str(lr) +"******************************************"
    learning_rate = lr

    n_train_batches = upsampled_x.shape[0]/batch_size
    n_valid_batches = up_val_x.shape[0]/batch_size
    n_test_batches = up_test_x.shape[0]/batch_size

    train_model,val_model,test_model = SRCNN_Theano.train_SRCNN(shared_x,shared_y,
                             shared_val_x,shared_val_y,
                             shared_test_x,shared_test_y,
                            n_train_batches, n_valid_batches, n_test_batches, 
                             n_epochs, batch_size,learning_rate,upsampling_factor=4)



