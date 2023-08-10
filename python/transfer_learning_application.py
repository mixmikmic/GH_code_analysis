#In Jupyter notebooks, you will need to run this command before doing any plotting
get_ipython().magic('matplotlib inline')
from sutils import *
import os, json
from glob import glob

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model,load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

limit_gpu_mem()

#path = "data/dogscats/"
path = "dataset/dogscats/sample/"

batch_size=8
no_of_epochs=10

# Prepare images for training in batches
# NB: They must be in subdirectories named based on their category
batches = get_batches(path+'train', batch_size=batch_size)
val_batches = get_batches(path+'valid', batch_size=batch_size*2)

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

base_model.summary()

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- one for dog and one for cat... 
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    print(layer.name)
    layer.trainable = False

for layer in model.layers:
    print(layer.name)
    print(layer.trainable)

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])




test = batches.next()

len(test[1][0])
test[1][2]

hist = model.fit_generator(batches, steps_per_epoch=8, epochs=no_of_epochs,verbose=1,
                validation_data=val_batches, validation_steps=3)

metrics = model.evaluate_generator(val_batches,10,10,workers=1,pickle_safe=False)
print("model accuracy:",metrics[1])

model.save('cats-dogs.hdf5')

model = load_model('cats-dogs.hdf5')

import os
cwd = os.getcwd()
print(cwd)
image_path = 'dataset/dogscats/test1/'

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import numpy as np

img_path = os.path.join(image_path, '31.jpg')
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print('Input image shape:', x.shape)



preds = model.predict(x)
print('Predicted:', preds)

import matplotlib.pyplot as plt
index = 25
plt.imshow(img)

result= np.argmax(preds)
if result==0:
    print("Its a cat")
else:
    print("Its a dog")    











