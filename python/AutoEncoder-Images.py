from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import numpy as np
from PIL import Image
import os, sys
from keras.callbacks import TensorBoard

image_folder = "/data/amnh/darwin/images_downsampled5x_7k"
all_img = os.listdir(image_folder)
all_crop = []
crop_size = 200

# import all images and pull a central crop normalized from 0-1
for filename in all_img:
    try: 
        im_path = os.path.join(image_folder,filename)
        im = Image.open(im_path)
        s = im.size
        im_crop = im.crop((s[0]/2-crop_size/2,s[1]/2-crop_size/2,s[0]/2+crop_size/2,s[1]/2+crop_size/2))
        all_crop.append(np.array(im_crop).astype('float32')/255)
    except:
        print("failed on {}".format(filename))

print("{} images loaded".format(len(all_crop)))

input_img = Input(shape=(crop_size, crop_size, 3))

# we start with a 200x200x3 central crop of the input image
x = Convolution2D(64, 3, 1, activation='relu', border_mode='same')(input_img)
x = Convolution2D(64, 1, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(32, 3, 1, activation='relu', border_mode='same')(x)
x = Convolution2D(32, 1, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(16, 3, 1, activation='relu', border_mode='same')(x)
x = Convolution2D(16, 1, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 1, activation='relu', border_mode='same')(x)
x = Convolution2D(8, 1, 3, activation='relu', border_mode='same')(x)

# we are at 8X8X8 ([200x200]/2/2/2/3 and 8 filters)
encoded = MaxPooling2D((5, 5), border_mode='same')(x)

x = Convolution2D(8, 1, 3, activation='relu', border_mode='same')(encoded)
x = Convolution2D(8, 3, 1, activation='relu', border_mode='same')(x)
x = UpSampling2D((5, 5))(x)
x = Convolution2D(16, 1, 3, activation='relu', border_mode='same')(x)
x = Convolution2D(16, 3, 1, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(32, 1, 3, activation='relu', border_mode='same')(x)
x = Convolution2D(32, 3, 1, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(64, 1, 3, activation='relu', border_mode='same')(x)
x = Convolution2D(64, 3, 1, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(3, 1, 3, activation='relu', border_mode='same')(x)

decoded = Convolution2D(3, 3, 1, activation='relu', border_mode='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# print layer shapes for debugging purposes
for i in autoencoder.layers:
    print(i.output_shape)

np.random.shuffle(all_crop)
training, test = np.array(all_crop[:int(.8*len(all_crop))]), np.array(all_crop[int(.8*len(all_crop)):])

autoencoder.fit(training, training,
                nb_epoch=10,
                batch_size=128,
                shuffle=True,
                validation_data=(test, test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

''' #NOT DELETING THIS JUST IN CASE!!! - marko
(i think can actually delete this) - marko

# we start with a 200x200pixel central crop of the input image
x = Convolution2D(3, 1, 64, activation='relu', border_mode='same')(input_img)
x = Convolution2D(1, 3, 64, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(3, 1, 32, activation='relu', border_mode='same')(x)
x = Convolution2D(1, 3, 32, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(3, 1, 16, activation='relu', border_mode='same')(x)
x = Convolution2D(1, 3, 16, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(3, 1, 8, activation='relu', border_mode='same')(x)
x = Convolution2D(1, 3, 8, activation='relu', border_mode='same')(x)

# we are at 8X8X8 ([200x200]/2/2/2/3 and 8 filters)
encoded = MaxPooling2D((5, 5), border_mode='same')(x)

x = Convolution2D(1, 3, 8, activation='relu', border_mode='same')(encoded)
x = Convolution2D(3, 1, 8, activation='relu', border_mode='same')(x)
x = UpSampling2D((5, 5))(x)
x = Convolution2D(1, 3, 16, activation='relu', border_mode='same')(x)
x = Convolution2D(3, 1, 16, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(1, 3, 32, activation='relu', border_mode='same')(x)
x = Convolution2D(3, 1, 32, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(1, 3, 64, activation='relu', border_mode='same')(x)
x = Convolution2D(3, 1, 64, activation='relu', border_mode='same')(x)
decoded = UpSampling2D((2, 2))(x)
#x = Convolution2D(3, 1, 1, activation='relu', border_mode='same')(x)
#decoded = Convolution2D(1, 3, 1, activation='relu', border_mode='same')(x)
'''

