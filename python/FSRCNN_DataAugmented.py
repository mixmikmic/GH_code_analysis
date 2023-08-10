get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
import theano
import theano.tensor as T
import numpy as np
import FSRCNN_Theano_Data
import os
import Fpreprocessing
from scipy import ndimage,misc
from PIL import Image

def get_image_prefix(image_name):
    return image_name.split('_', 1)[0]
def get_image_width(image_name):
    yo = image_name.split("_")
    return yo[2]
def get_image_height(image_name):
    yo = image_name.split("_")
    return yo[3].split('.',1)[0]
def create_image(image_folder, output_folder, output_type = 'YCbCr',upsampling_factor = 4):
    reconstruct = dict()
    for (dirpath,dirnames,filenames) in os.walk(image_folder):
        print image_folder
        filenames.sort()
        #print(filenames)
        
        for counter,image_filename in enumerate(filenames):
            print(get_image_prefix(image_filename))
            print('\n')
            actual_image_name = get_image_prefix(image_filename)
            if(actual_image_name not in reconstruct):
                reconstruct[actual_image_name] = np.zeros((8,8,3))
            if image_filename.split('.')[-1] == 'bmp' and image_filename[0] != '.':
                if counter % 10 == 0:
                    print "processed:" + str(counter)
                image = misc.imread(os.path.join(image_folder,image_filename),flatten=False, mode = output_type)
                #(width,height,channel_depth)
                w = int(get_image_width(image_filename))
                h = int(get_image_height(image_filename))
                print (w,h)
                print(image.shape)
                print(reconstruct[actual_image_name][14*w:14*w+33,14*h:14*h+33,:].shape)
                reconstruct[actual_image_name][14*w:14*w+33,14*h:14*h+33,:] = image
                #misc.imshow(reconstruct[actual_image_name])
                #img = Image.fromarray(data, 'RGB')
                img = Image.fromarray(reconstruct[actual_image_name], 'RGB')
                img.save('my.png')
                img.show()
                misc.imsave(os.path.join(actual_image_name+'.bmp'),reconstruct[actual_image_name])
    #for key in recstruct:
        #misc.imsave(os.path.join(output_folder,actual_image_name+'.bmp'),reconstruct[actual_image_name])
#create_image('/home/ubuntu/Data/Validation_Subsamples_RGB_4','/home/ubuntu/Data/Reconstructed',output_type ='RGB',upsampling_factor = 4)

#Create Validation_Subsamples_RGB_4 and Validation_Subsamples_RGB_4_GT folders
'''
Fpreprocessing.create_subimages('/home/ubuntu/Data/Set5',
'/home/ubuntu/Data/Validation_Subsamples_RGB_4/',
output_type ='RGB',upsampling_factor = 4)

Fpreprocessing.create_subimages('/home/ubuntu/Data/Training_Full',
'/home/ubuntu/Data/Training_Subsamples_RGB_4/',
output_type ='RGB',upsampling_factor = 4)
'''

'''
Fpreprocessing.create_subimages('/home/ubuntu/Data/Set14',
'/home/ubuntu/Data/Test_Subsamples_RGB_4/',
output_type ='RGB',upsampling_factor = 4)
'''

import FSRCNN_Theano

#load dataset

#load training
data_x = FSRCNN_Theano_Data.load_dataset('/home/ubuntu/Data/Training_Subsamples_RGB_4','data_x')
data_y = FSRCNN_Theano_Data.load_dataset('/home/ubuntu/Data/Training_Subsamples_RGB_4_gt','data_y')
valid_x = FSRCNN_Theano_Data.load_dataset('/home/ubuntu/Data/Validation_Subsamples_RGB_4','data_x')
valid_y = FSRCNN_Theano_Data.load_dataset('/home/ubuntu/Data/Validation_Subsamples_RGB_4_gt','data_y')
test_x = FSRCNN_Theano_Data.load_dataset('/home/ubuntu/Data/Test_Subsamples_RGB_4','data_x')
test_y = FSRCNN_Theano_Data.load_dataset('/home/ubuntu/Data/Test_Subsamples_RGB_4_gt','data_y')
print "done loading\n\n"
print "data_x: " + str(data_x.shape)
print "data_y: " + str(data_y.shape)
print "valid_x: " + str(valid_x.shape)
print "valid_y: " + str(valid_y.shape)
print "test_x: " + str(test_x.shape)
print "test_y: " + str(test_y.shape)

#Bicubic interp to save computation during training
upsampled_x = data_x #Fpreprocessing.upsample(data_x) #33,33,3 input images expected
up_val_x = valid_x #Fpreprocessing.upsample(valid_x) #33,33,3 input images expected
up_test_x = test_x #Fpreprocessing.upsample(test_x) #33,33,3 input images expected

#Reshape for training,valid,test

print upsampled_x.shape
print data_y.shape
print up_val_x.shape
print up_test_x.shape

upsampled_x = upsampled_x.reshape((22092,8*8*3))
data_y = data_y.reshape((22092,33*33*3))
up_val_x = up_val_x.reshape((2488,8*8*3))
valid_y = valid_y.reshape((2488,33*33*3))
up_test_x = up_test_x.reshape((14851,8*8*3))
test_y = test_y.reshape((14851,33*33*3))

print upsampled_x.shape
print data_y.shape
print up_val_x.shape
print up_test_x.shape



get_ipython().magic('autosave 300')

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
batch_size = 50
train_set_x = shared_x
n_epochs = 100
lrs = [.0005]
for lr in lrs:
    print "\n\n ****************************** lr = " + str(lr) +"******************************************"
    learning_rate = lr

    n_train_batches = upsampled_x.shape[0]/batch_size
    n_valid_batches = up_val_x.shape[0]/batch_size
    n_test_batches = up_test_x.shape[0]/batch_size



    val_model,test_model = FSRCNN_Theano_Data.train_FSRCNN(shared_x,shared_y,
                             shared_val_x,shared_val_y,
                             shared_test_x,shared_test_y,
                            n_train_batches, n_valid_batches, n_test_batches, 
                             n_epochs, batch_size,learning_rate,upsampling_factor=4, flip_p=0.5, translate_p=0.15, rotate_p=0.15)
reconstructed_imgs = np.zeros(((n_valid_batches+1)*batch_size, 3, 17, 17))
for i in xrange(n_valid_batches):
     cost,MSE_per_pixel,psnr,reconstucted_patches = val_model(i)
     reconstructed_imgs[i*batch_size:(i+1)*batch_size,:,:,:] = reconstucted_patches

FSRCNN_Theano.rebuild_images(reconstructed_imgs,'/home/ubuntu/Data/Validation_Subsamples_RGB_4/',patch_dim=17,dataset='validate_lr=5e4_batch=50_data_augment')
reconstructed_imgs = np.zeros((14851, 3, 17, 17))
for i in xrange(n_test_batches):
     cost,MSE_per_pixel,psnr,reconstucted_patches = test_model(i)
     reconstructed_imgs[i*batch_size:(i+1)*batch_size,:,:,:] = reconstucted_patches

FSRCNN_Theano.rebuild_images(reconstructed_imgs,'/home/ubuntu/Data/Test_Subsamples_RGB_4/',patch_dim=17,dataset='test_lr=5e4_batch_data_augment')

reconstructed_imgs = np.zeros(((n_valid_batches+1)*batch_size, 3, 17, 17))
for i in xrange(n_valid_batches):
     cost,MSE_per_pixel,psnr,reconstucted_patches = val_model(i)
     reconstructed_imgs[i*batch_size:(i+1)*batch_size,:,:,:] = reconstucted_patches

FSRCNN_Theano.rebuild_images(reconstructed_imgs,'/home/ubuntu/Data/Validation_Subsamples_RGB_4/',patch_dim=17,dataset='validate_lr=5e4_batch=50_data_augment', place=True)
reconstructed_imgs = np.zeros((14851, 3, 17, 17))
for i in xrange(n_test_batches):
     cost,MSE_per_pixel,psnr,reconstucted_patches = test_model(i)
     reconstructed_imgs[i*batch_size:(i+1)*batch_size,:,:,:] = reconstucted_patches

FSRCNN_Theano.rebuild_images(reconstructed_imgs,'/home/ubuntu/Data/Test_Subsamples_RGB_4/',patch_dim=17,dataset='test_lr=5e4_batch=50_data_augment',place=True)

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
batch_size = 25
train_set_x = shared_x
n_epochs = 100
lrs = [.0005]
for lr in lrs:
    print "\n\n ****************************** lr = " + str(lr) +"******************************************"
    learning_rate = lr

    n_train_batches = upsampled_x.shape[0]/batch_size
    n_valid_batches = up_val_x.shape[0]/batch_size
    n_test_batches = up_test_x.shape[0]/batch_size



    val_model,test_model = FSRCNN_Theano_Data.train_FSRCNN(shared_x,shared_y,
                             shared_val_x,shared_val_y,
                             shared_test_x,shared_test_y,
                            n_train_batches, n_valid_batches, n_test_batches, 
                             n_epochs, batch_size,learning_rate,upsampling_factor=4, flip_p=0.5, translate_p=0.15, rotate_p=0.15)
reconstructed_imgs = np.zeros(((n_valid_batches+1)*batch_size, 3, 17, 17))
for i in xrange(n_valid_batches):
     cost,MSE_per_pixel,psnr,reconstucted_patches = val_model(i)
     reconstructed_imgs[i*batch_size:(i+1)*batch_size,:,:,:] = reconstucted_patches

FSRCNN_Theano.rebuild_images(reconstructed_imgs,'/home/ubuntu/Data/Validation_Subsamples_RGB_4/',patch_dim=17,dataset='validate_lr=5e4_batch=50_data_augment')
reconstructed_imgs = np.zeros((14851, 3, 17, 17))
for i in xrange(n_test_batches):
     cost,MSE_per_pixel,psnr,reconstucted_patches = test_model(i)
     reconstructed_imgs[i*batch_size:(i+1)*batch_size,:,:,:] = reconstucted_patches

FSRCNN_Theano.rebuild_images(reconstructed_imgs,'/home/ubuntu/Data/Test_Subsamples_RGB_4/',patch_dim=17,dataset='test_lr=5e4_batch_data_augment')

reconstructed_imgs = np.zeros(((n_valid_batches+1)*batch_size, 3, 17, 17))
for i in xrange(n_valid_batches):
     cost,MSE_per_pixel,psnr,reconstucted_patches = val_model(i)
     reconstructed_imgs[i*batch_size:(i+1)*batch_size,:,:,:] = reconstucted_patches

FSRCNN_Theano.rebuild_images(reconstructed_imgs,'/home/ubuntu/Data/Validation_Subsamples_RGB_4/',patch_dim=17,dataset='validate_lr=5e4_batch=50_data_augment', place=True)
reconstructed_imgs = np.zeros((14851, 3, 17, 17))
for i in xrange(n_test_batches):
     cost,MSE_per_pixel,psnr,reconstucted_patches = test_model(i)
     reconstructed_imgs[i*batch_size:(i+1)*batch_size,:,:,:] = reconstucted_patches

FSRCNN_Theano.rebuild_images(reconstructed_imgs,'/home/ubuntu/Data/Test_Subsamples_RGB_4/',patch_dim=17,dataset='test_lr=5e4_batch=50_data_augment',place=True)

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
batch_size = 25
train_set_x = shared_x
n_epochs = 100
lrs = [.0005]
for lr in lrs:
    print "\n\n ****************************** lr = " + str(lr) +"******************************************"
    learning_rate = lr

    n_train_batches = upsampled_x.shape[0]/batch_size
    n_valid_batches = up_val_x.shape[0]/batch_size
    n_test_batches = up_test_x.shape[0]/batch_size



    val_model,test_model = FSRCNN_Theano_Data.train_FSRCNN(shared_x,shared_y,
                             shared_val_x,shared_val_y,
                             shared_test_x,shared_test_y,
                            n_train_batches, n_valid_batches, n_test_batches, 
                             n_epochs, batch_size,learning_rate,upsampling_factor=4, flip_p=0.5, translate_p=0, rotate_p=0)
reconstructed_imgs = np.zeros(((n_valid_batches+1)*batch_size, 3, 17, 17))
for i in xrange(n_valid_batches):
     cost,MSE_per_pixel,psnr,reconstucted_patches = val_model(i)
     reconstructed_imgs[i*batch_size:(i+1)*batch_size,:,:,:] = reconstucted_patches

FSRCNN_Theano.rebuild_images(reconstructed_imgs,'/home/ubuntu/Data/Validation_Subsamples_RGB_4/',patch_dim=17,dataset='validate_lr=5e4_batch=50_data_augment')
reconstructed_imgs = np.zeros((14851, 3, 17, 17))
for i in xrange(n_test_batches):
     cost,MSE_per_pixel,psnr,reconstucted_patches = test_model(i)
     reconstructed_imgs[i*batch_size:(i+1)*batch_size,:,:,:] = reconstucted_patches

FSRCNN_Theano.rebuild_images(reconstructed_imgs,'/home/ubuntu/Data/Test_Subsamples_RGB_4/',patch_dim=17,dataset='test_lr=5e4_batch_data_augment')

reconstructed_imgs = np.zeros(((n_valid_batches+1)*batch_size, 3, 17, 17))
for i in xrange(n_valid_batches):
     cost,MSE_per_pixel,psnr,reconstucted_patches = val_model(i)
     reconstructed_imgs[i*batch_size:(i+1)*batch_size,:,:,:] = reconstucted_patches

FSRCNN_Theano.rebuild_images(reconstructed_imgs,'/home/ubuntu/Data/Validation_Subsamples_RGB_4/',patch_dim=17,dataset='validate_lr=5e4_batch=50_data_augment', place=True)
reconstructed_imgs = np.zeros((14851, 3, 17, 17))
for i in xrange(n_test_batches):
     cost,MSE_per_pixel,psnr,reconstucted_patches = test_model(i)
     reconstructed_imgs[i*batch_size:(i+1)*batch_size,:,:,:] = reconstucted_patches

FSRCNN_Theano.rebuild_images(reconstructed_imgs,'/home/ubuntu/Data/Test_Subsamples_RGB_4/',patch_dim=17,dataset='test_lr=5e4_batch=50_data_augment',place=True)

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
batch_size = 50
train_set_x = shared_x
n_epochs = 50
lrs = [.0005]
for lr in lrs:
    print "\n\n ****************************** lr = " + str(lr) +"******************************************"
    learning_rate = lr

    n_train_batches = upsampled_x.shape[0]/batch_size
    n_valid_batches = up_val_x.shape[0]/batch_size
    n_test_batches = up_test_x.shape[0]/batch_size



    val_model,test_model = FSRCNN_Theano_Data.train_FSRCNN(shared_x,shared_y,
                             shared_val_x,shared_val_y,
                             shared_test_x,shared_test_y,
                            n_train_batches, n_valid_batches, n_test_batches, 
                             n_epochs, batch_size,learning_rate,upsampling_factor=4, flip_p=0.5, translate_p=0.35, rotate_p=0)
reconstructed_imgs = np.zeros(((n_valid_batches+1)*batch_size, 3, 17, 17))
for i in xrange(n_valid_batches):
     cost,MSE_per_pixel,psnr,reconstucted_patches = val_model(i)
     reconstructed_imgs[i*batch_size:(i+1)*batch_size,:,:,:] = reconstucted_patches

FSRCNN_Theano.rebuild_images(reconstructed_imgs,'/home/ubuntu/Data/Validation_Subsamples_RGB_4/',patch_dim=17,dataset='validate_lr=5e4_batch=50_data_augment_translate')
reconstructed_imgs = np.zeros((14851, 3, 17, 17))
for i in xrange(n_test_batches):
     cost,MSE_per_pixel,psnr,reconstucted_patches = test_model(i)
     reconstructed_imgs[i*batch_size:(i+1)*batch_size,:,:,:] = reconstucted_patches

FSRCNN_Theano.rebuild_images(reconstructed_imgs,'/home/ubuntu/Data/Test_Subsamples_RGB_4/',patch_dim=17,dataset='test_lr=5e4_batch_data_augment_translate')

reconstructed_imgs = np.zeros(((n_valid_batches+1)*batch_size, 3, 17, 17))
for i in xrange(n_valid_batches):
     cost,MSE_per_pixel,psnr,reconstucted_patches = val_model(i)
     reconstructed_imgs[i*batch_size:(i+1)*batch_size,:,:,:] = reconstucted_patches

FSRCNN_Theano.rebuild_images(reconstructed_imgs,'/home/ubuntu/Data/Validation_Subsamples_RGB_4/',patch_dim=17,dataset='validate_lr=5e4_batch=50_data_augment_translate', place=True)
reconstructed_imgs = np.zeros((14851, 3, 17, 17))
for i in xrange(n_test_batches):
     cost,MSE_per_pixel,psnr,reconstucted_patches = test_model(i)
     reconstructed_imgs[i*batch_size:(i+1)*batch_size,:,:,:] = reconstucted_patches

FSRCNN_Theano.rebuild_images(reconstructed_imgs,'/home/ubuntu/Data/Test_Subsamples_RGB_4/',patch_dim=17,dataset='test_lr=5e4_batch=50_data_augment_translate',place=True)

