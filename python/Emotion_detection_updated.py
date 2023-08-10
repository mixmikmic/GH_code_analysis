import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras

from os import listdir
path = "E:\Datasets\KDEF_and_AKDEF\\KDEF"
pathlist = listdir(path)

from shutil import copyfile
import os

for i in range(len(pathlist)):
    img_path = listdir(path+"\\"+pathlist[i])
    for j in range(len(img_path)):
        if(img_path[j][4:6]=="AF"):
            dst = "E:\\Datasets\\KDEF_and_AKDEF\\data\\AF"
        elif(img_path[j][4:6]=="AN"):
            dst = "E:\\Datasets\\KDEF_and_AKDEF\\data\\AN"
        elif(img_path[j][4:6]=="DI"):
            dst = "E:\\Datasets\\KDEF_and_AKDEF\\data\\DI"
        elif(img_path[j][4:6]=="HA"):
            dst = "E:\\Datasets\\KDEF_and_AKDEF\\data\\HA"
        elif(img_path[j][4:6]=="NE"):
            dst = "E:\\Datasets\\KDEF_and_AKDEF\\data\\NE"
        elif(img_path[j][4:6]=="SA"):
            dst = "E:\\Datasets\\KDEF_and_AKDEF\\data\\SA"
        elif(img_path[j][4:6]=="SU"):
            dst = "E:\\Datasets\\KDEF_and_AKDEF\\data\\SU"
        
        if not os.path.exists(dst):
            os.makedirs(dst)
        
        src = path+"\\"+pathlist[i]+"\\"+img_path[j]
        dst = dst + "\\" + img_path[j]
        try:
            copyfile(src, dst)
        except:
            pass

from keras.datasets import mnist
from keras.utils import to_categorical

from keras import models
from keras import layers

from keras import losses, optimizers, metrics

from keras.preprocessing.image import ImageDataGenerator
train_dir = '/media/shuvendu/Projects/Datasets/KDEF_and_AKDEF/data/train'
test_dir ='/media/shuvendu/Projects/Datasets/KDEF_and_AKDEF/data/test'


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (150, 150),
    batch_size=32
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = (150, 150),
    batch_size=32
)

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))

model.summary()

model.compile(optimizer=optimizers.RMSprop(),
              loss=losses.categorical_crossentropy,
              metrics=['accuracy']
             )

history = model.fit_generator(
    train_generator,
    steps_per_epoch = 155,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=10
)

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training losses')
plt.plot(epochs, val_loss, 'b', label='Validation losses')
plt.title('Training and validation loss')
plt.legend()

plt.show()

from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                 include_top=False,
                 input_shape=(150, 150, 3))

model = models.Sequential()

model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(512))
model.add(layers.Dense(7, activation='softmax'))
model.summary()

conv_base.trainable = False
print('This is the number of trainable weights before freezing the conv base:', len(model.trainable_weights))

model.compile(
    optimizer=optimizers.Adam(lr=5e-5), 
    loss=losses.categorical_crossentropy, 
    metrics=[metrics.categorical_accuracy]
)

history = model.fit_generator(train_generator, 
                              steps_per_epoch=155, 
                              epochs=15, 
                              validation_data=validation_generator, 
                              validation_steps=10)

from keras.models import load_model
model = load_model('fc_pretrain.h5')
conv_base = model.layers[0]

model.summary()

conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(
    optimizer=optimizers.RMSprop(lr=1e-4), 
    loss=losses.categorical_crossentropy, 
    metrics=[metrics.categorical_accuracy]
)

history = model.fit_generator(train_generator, 
                              steps_per_epoch=125, 
                              epochs=15, 
                              validation_data=validation_generator, 
                              validation_steps=30)

import matplotlib.pyplot as plt

acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training losses')
plt.plot(epochs, val_loss, 'b', label='Validation losses')
plt.title('Training and validation loss')
plt.legend()

plt.show()

model.save('fully_trained_model.h5')

from keras.models import load_model
model = load_model('fully_trained_model.h5')

import cv2
import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')

labels = {
    0 : "Afraid",
    1 : "Angry",
    2 : "disgust",
    3 : "happy",
    4 : "neutral",
    5 : "sad",
    6 : "surprised"
}

image = cv2.imread('E:\\2.jpg')
image.shape

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

image=np.array(cv2.resize(image, (150,150)))
image = image.reshape(1, 150, 150, 3)
image.shape

labels[np.argmax(model.predict(image))]



