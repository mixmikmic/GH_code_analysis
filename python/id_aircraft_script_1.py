import logging
import numpy as np
import os

np.random.seed(371250) # For reproducibility, needs to be set before Keras is loaded
from imp import reload
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

reload(logging)
logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%H:%M:%S")

#img_width, img_height = 256, 170 # Approximately 25% of the original
img_width, img_height = 150, 150

train_data_dir = "../data/train"
validation_data_dir = "../data/test"

labels = ["A321", "A340", "B747", "CRJ900"]
nb_train_samples = 3700 # actual 3726
nb_validation_samples = 630
nb_epoch = 20

logging.info("Current PID : {}".format(os.getpid()))

model_input_shape = (3, img_width, img_height)

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(3, img_width, img_height)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(len(labels)))
model.add(Activation("sigmoid"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

train_datagen = ImageDataGenerator(
    fill_mode="nearest",
    horizontal_flip=True,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    batch_size=32,
    classes=labels,
    target_size=(img_width, img_height),
    class_mode="categorical")

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    batch_size=32,
    classes=labels,
    target_size=(img_width, img_height),
    class_mode="categorical")

model.fit_generator(
    train_generator,
    nb_epoch=nb_epoch,
    nb_val_samples=nb_validation_samples,
    samples_per_epoch=nb_train_samples,
    validation_data=validation_generator,
    verbose=2)
model.save_weights("model_1_weights_{}_epochs.h5".format(nb_epoch))
logging.info("Done")

