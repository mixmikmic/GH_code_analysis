from keras import applications

get_ipython().system('curl "https://00e9e64bac4eee7d3c2139382ac60caacc0ec0101dd450cfc7-apidata.googleusercontent.com/download/storage/v1/b/kagglecatdog/o/train%2Fcat.1000.jpg?qk=AD5uMEtsLAxVoY_qsxCYzOG4GsoM4HgCzvwAZSbzvo9vRHgByyYfCXsCSHMlfqnl4BK-ZwTW6_gluVsoI558jwKuWIe9gfMRPoA4GIGCAVEadosUugpAszdutTLZwDAkz5hSnVz9qbdLaMXEPWlwU7VXTCwCAnrXBzA9TFAwMR9DROcmojktsT_7CVPx8ZvucPbBm6R5oL7KCk-EAePXf44vYii_nuOsvTOiUa6fuWuT1UkbhgB7zD-otPkmsYwbK0tGxVn3SohyhjQp9CstHMJA3Gh9xD7uT7dePVZkyeh8reClodzD1mivo_PajgKboh2auW_2SAzfe5xU2G06MUiiYtBvZygFkrq7Ofk5myNUWGa_GIoidz3jjL3UWWVcMZ0F8CTH16CpxysEUxPHovlch1J028dnWm74VT0__hEHZg8f8og-SyURDqoc_6G8wYanAnEvxWGdz8dhWoFX8UD1GlD9ZSDgBjmw0CB3g2HxQeTUiCf5dPsd_Z9xzNqSuhiSUcEyvxAhdLdvCXWW8YeOvrHU7OjF8dFl3cJ8sG-G8VblyCtFnG4eRSEp8Erfk_-xOTpFZmggJIQ-8LSQR_HxsF513aiPKRnjYG6XMfCC6vjhIN94yke2zC93AFDbLAoFloEDHMUFAUbyG7a41MXFZp3uFAeD1MYyB3DncMNcfwHzL168o1O9zrcDY94XuoT42XEZsfL3BB-o9KDEGK0lHhDl1x7zZS-sx0PIvvKFw3IKlm41q7o9jEB7kTGMe0cG6A2qK4fWvEVORHd_kaGLO6sZPH0Ciw" > ../data/cat14.jpg')

from skimage import io, img_as_float

from skimage.transform import rescale, resize, downscale_local_mean

image_temp = io.imread('../data/cat1.jpg')

image_temp.shape

get_ipython().magic('pinfo2 resize')

image_resize = resize(image_temp, (256,256), mode = "reflect")

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

plt.imshow(image_resize)

image_resize.shape

height, width, channels = image_resize.shape

height

width

channels

import os
import shutil
import glob
import re

# Image processing pipeline
def image_processing(img_src, img_dest):
    
    img_list = glob.glob(img_src+'/*jpg')
    
    for img in img_list:
        image_temp = io.imread(img)
        image_temp = resize(image_temp, (256,256), mode = "reflect")
        image_temp = img_as_float(image_temp)
        # since the image is already normalized, no need for normalization
        height, width, channels = image_temp.shape
        print (height, ":", width, ":", channels)

        if not(os.path.exists(img_dest+'preprocessed/')):
            os.mkdir(img_dest+'preprocessed')
        else:
            print("The Path exists")
        
        print(img)
        print(re.split('/',img)[2])
        io.imsave(img_dest+'preprocessed/'+str(re.split('/',img)[2]), image_temp)
        print("The file {} has been saved.".format(str(re.split('/',img)[2])))

image_processing('../data/', '../data/')

img = '../data/cat12.jpg'

if re.match(r'([A-z0-9\.]+(.jpg))', str(img)):
    print("yes")

get_ipython().magic('pinfo2 io.imsave')

from keras.models import Model, Sequential
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D

import h5py

img_width = 256
img_height = 256
model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

model.summary()

model.layers

for layer in model.layers[:10]:
    layer.trainable = False

#Adding custom Layers 
x = model.output
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(16, activation="softmax")(x)

# creating the final model 
model_final = Model(input = model.input, output = predictions)

# compile the model 
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

def batch_processing(train_src, validation_src, img_width, img_height, non_train_layers, batch_size, epochs):
    model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    
    for layer in model.layers[:non_train_layers]:
        layer.trainable = False
        
    #Adding custom Layers 
    x = model.output
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(16, activation="softmax")(x)
    
    # creating the final model 
    model_final = Model(input = model.input, output = predictions)
    
    # compile the model 
    model_final.compile(loss = "categorical_crossentropy", optimizer =                         optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
            #'data/train',  # this is the target directory
            train_src,
            target_size=(256, 256),  # all images will be resized to 256x256
            batch_size=batch_size,
            class_mode='categorical')  

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            validation_src,
            target_size=(256, 256),
            batch_size=batch_size,
            class_mode='categorical')
    
    model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
    
    model.save_weights('first_try.h5')  # always save your weights after training or during training

    



