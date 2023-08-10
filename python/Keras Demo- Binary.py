import keras
import numpy as np
import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import SGD, Adam
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib figure size and make inline
matplotlib.rcParams['figure.figsize'] = [10, 6]
get_ipython().run_line_magic('matplotlib', 'inline')

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test  = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')

# normalize the pixel values to be in the range (0, 1)
x_train /= 255 
x_test  /= 255


print("Number of training samples in full dataset: " + str(x_train.shape[0]))
print("Number of testing samples in full dataset: " + str(x_test.shape[0]))

# Create a binary version for binary classification
df = pd.DataFrame(np.vstack([x_train, x_test]))
df['label'] = pd.Series(np.concatenate([y_train, y_test]))
binary_data = df[df['label'].isin([0, 1])]

# Create the binary dataset
binary_x_train = binary_data.iloc[:10000].drop(['label'], axis=1).as_matrix()
binary_y_train = binary_data.iloc[:10000]['label'].as_matrix()

binary_x_validation = binary_data.iloc[10000:11000].drop(['label'], axis=1).as_matrix()
binary_y_validation = binary_data.iloc[10000:11000]['label'].as_matrix()

binary_x_test  = binary_data.iloc[11000:].drop(['label'], axis=1).as_matrix()
binary_y_test  = binary_data.iloc[11000:]['label'].as_matrix()

print()
print("Total number of binary 0 / 1 examples: ", binary_data.shape)
print("Number of training samples in BINARY dataset: " + str(binary_x_train.shape[0]))
print("Number of testing samples in BINARY dataset: " + str(binary_x_test.shape[0]))
print("Number of validation samples in BINARY dataset: " + str(binary_x_validation.shape[0]))

image = binary_x_train[0]
img_rows, img_cols, channels = 28, 28, 1
image = np.reshape(image, [img_rows, img_cols])
plt.imshow(image, cmap='gray')
plt.axis('off')

# The shape of each image is a vector with 784 binary values ("pixels")
image_shape = binary_x_train[0].shape
print("The shape of each image is" + str(image_shape))

binary_model = Sequential()
num_classes = 1
binary_model.add(Dense(num_classes, activation='sigmoid', input_shape=image_shape))

binary_model.compile(loss='binary_crossentropy', optimizer=SGD(), metrics=['accuracy'])

global_batch_size = 128
num_epochs = 10

history = binary_model.fit(binary_x_train, binary_y_train,
                   batch_size=global_batch_size, # average 128 examples in each SGD test
                   epochs=num_epochs,
                   verbose=1,
                   validation_data=(binary_x_validation, binary_y_validation)) 

score = binary_model.evaluate(binary_x_test, binary_y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

