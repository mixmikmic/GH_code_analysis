import os
from shutil import copyfile
import numpy as np

# take a look at the data
i = 0
for f in os.listdir('train'):
    print f
    i += 1
    if i > 10:
        break

# prepare training set (1000 images per class)
os.makedirs('data/train')
os.makedirs('data/train/cats')
os.makedirs('data/train/dogs')

for i in range(1000):
    copyfile('train/cat.%d.jpg' % i, 'data/train/cats/cat.%d.jpg' % i)
    copyfile('train/dog.%d.jpg' % i, 'data/train/dogs/dog.%d.jpg' % i)

# prepare validation set (400 images per class by random sampling)
sample_indices = np.random.choice(range(1000, 12500), 400, replace=False)

print len(sample_indices)
print len(np.unique(sample_indices))

# prepare validation set (400 images per class)
os.makedirs('data/validation')
os.makedirs('data/validation/cats')
os.makedirs('data/validation/dogs')

for i in sample_indices:
    copyfile('train/cat.%d.jpg' % i, 'data/validation/cats/cat.%d.jpg' % i)
    copyfile('train/dog.%d.jpg' % i, 'data/validation/dogs/dog.%d.jpg' % i)

get_ipython().system(' tree -d data')

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')

# try to generate 15 images
raw_img = load_img('data/train/cats/cat.78.jpg')
x = img_to_array(raw_img)
print x.shape

# convert to tensor
x = x.reshape((1, ) + x.shape)
print x.shape

i = 0
gen_images = []
for batch in datagen.flow(x, batch_size=1):
    gen_images.append(batch)
    i += 1
    if i > 15:
        # otherwise the generator would loop indefinitely
        break

print len(gen_images)
print gen_images[0].shape

# show the generate images
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
get_ipython().magic('matplotlib inline')

fig = plt.figure(1, (16, 16))
grids = ImageGrid(fig, 111, nrows_ncols = (4, 4), axes_pad=0)

grids[0].imshow(raw_img)
grids[0].text(0.5, 0.05, 'Original Image', verticalalignment='bottom', horizontalalignment='center',
              transform=grids[0].transAxes, color='white', fontsize=12, bbox={'facecolor':'black', 'pad': 5})

for i in range(1, 16):
    grids[i].imshow(array_to_img(gen_images[i-1][0]))
    grids[i].text(0.5, 0.05, 'Augmented %d' %(i), verticalalignment='bottom', horizontalalignment='center',
                  transform=grids[i].transAxes, color='white', fontsize=12, bbox={'facecolor':'black', 'pad': 5})
    grids[i].set_xticks([])
    grids[i].set_yticks([])
    
plt.show()

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)

# It should contain one subdirectory per class. 
# Any PNG, JPG, BMP, PPM or TIF images inside each of the 
# subdirectories directory tree will be included in the generator.
train_generator = train_datagen.flow_from_directory('data/train', 
                                                    target_size=(150, 150), 
                                                    batch_size=batch_size, 
                                                    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory('data/validation', 
                                                              target_size=(150, 150), 
                                                              batch_size=batch_size, 
                                                              class_mode='binary')

model.fit_generator(train_generator, 
                    steps_per_epoch=2000 // batch_size, 
                    epochs=50, 
                    validation_data=validation_generator, 
                    validation_steps=800 // batch_size)
model.save_weights('baseline.h5')

from keras import applications

# save_bottlebeck_features
datagen = ImageDataGenerator(rescale=1. / 255)

# build the VGG16 network
model = applications.VGG16(include_top=False, weights='imagenet')

batch_size=32
train_generator = datagen.flow_from_directory('data/train', target_size=(150, 150), 
                                              batch_size=batch_size, class_mode=None, shuffle=False)
# this means our generator will only yield batches of data, no labels
# our data will be in order, so all first 1000 images will be cats, then 1000 dogs

bottleneck_features_train = model.predict_generator(train_generator, int(np.ceil(2000. / batch_size)))
np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)

validation_generator = datagen.flow_from_directory('data/validation', target_size=(150, 150),
                                                   batch_size=batch_size, class_mode=None, shuffle=False)
bottleneck_features_validation = model.predict_generator(validation_generator, int(np.ceil(800. / batch_size)))
np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)

train_data = np.load(open('bottleneck_features_train.npy'))
# the features were saved in order, so recreating the labels is easy
train_labels = np.array([0] * 1000 + [1] * 1000)

validation_data = np.load(open('bottleneck_features_validation.npy'))
validation_labels = np.array([0] * 400 + [1] * 400)

train_data.shape

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_data, train_labels, epochs=50, batch_size=batch_size,
          validation_data=(validation_data, validation_labels))
model.save_weights('bottleneck_fc_model.h5')

# path to the model weights files.
weights_path = '../keras/examples/vgg16_weights.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'
# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 32

# build the VGG16 network
vgg_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
print('Model loaded.')

vgg_model.output_shape

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

from keras.models import Model
# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
model = Model(inputs= vgg_model.input, outputs= top_model(vgg_model.output))

model.summary()

from keras import optimizers

# set the first 15 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:15]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# fine-tune the model
model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)

