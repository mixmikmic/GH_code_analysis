# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
# Convolution operator for filtering windows of two-dimensional inputs.
# For the first layer in the model,use argument, input_shape=(64, 64, 3)`for 64x64 RGB pictures with Tensorflow backend
# 32,3,3 means - Apply a 3x3 convolution with 32 output filters

classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
# Max pooling operation to capture spatial features
# pool_size is the stride window size
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
# Input dim is not required for Convolution2D as this is not the first layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
# Binary cross entropy is used as there are only 2 classes - cats & dogs
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# ImageDataGenerator in Keras is used for Image Augmentation

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# Flow from Directory expects the data to be organized in a certain way: Root folder having subfolders for each class
training_set = train_datagen.flow_from_directory('0.datasets/catsdogs/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('0.datasets/catsdogs/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 3,  # No.of rounds. Higher number = Higher Accuracy but more time to train
                         validation_data = test_set,
                         nb_val_samples = 2000)

import numpy as np
import pandas as pd
from keras.preprocessing import image

to_predict = pd.DataFrame()
to_predict['Images'] = ''
to_predict['Prediction'] = ''
predictions=[]
#print(training_set.class_indices)

to_predict_image_path = '0.datasets/catsdogs/to_predict_set/'

for i in range(10):  # 10 images are present in the to-predict folder
    x = str(i+1)
    img_name = 'Img' + x + '.jpg'
    to_predict_image_string = to_predict_image_path + img_name 
    temp_image = image.load_img(to_predict_image_string,target_size = (64, 64))
    temp_image = image.img_to_array(temp_image)
    temp_image = np.expand_dims(temp_image,axis = 0)
    result = classifier.predict_classes(temp_image)
    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'
    
    to_predict.at[i, 'Images'] = img_name
    to_predict.at[i, 'Prediction'] = prediction
    predictions.append(prediction)  

# Final prediction
print(to_predict)

