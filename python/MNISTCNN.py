from keras.datasets import mnist

get_ipython().magic('matplotlib inline')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import keras
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D

num_classes = 10

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# input image dimensions
img_rows, img_cols = 28, 28

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

from keras.layers import Dropout

model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

model.summary()

from keras.callbacks import TensorBoard

model.compile(
    loss='categorical_crossentropy', 
    optimizer='adadelta', 
    metrics=['accuracy'])

model.fit(
    x_train, 
    y_train, 
    epochs=12, 
    batch_size=50, 
    validation_split=0.1,
    # ignore this line for now
    callbacks=[TensorBoard(histogram_freq=1, log_dir='mnist_cnn_logs')],
    verbose=2)

model.save('models/mnist_conv.h5')

model = load_model('models/mnist_conv.h5')

W = model.layers[0].get_weights()[0]

for i in range(32):
    plt.subplot(6, 6, i+1)
    weight = W[:, :, 0, i]
    plt.imshow(weight, cmap='gray')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)

from keras import backend as K

first_conv = model.layers[0]

first_conv_activation_fxn = K.function([model.input], [first_conv.output])

plt.imshow(x_test[0][:, :, 0])

first_conv_activation = first_conv_activation_fxn([[x_test[0]]])

for i in range(32):
    plt.subplot(6, 6, i+1)
    act = first_conv_activation[0][0, :, :, i]
    plt.imshow(act)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)

second_conv = model.layers[1]

second_conv_activation_fxn = K.function([model.input], [second_conv.output])

second_conv_activation = second_conv_activation_fxn([[x_test[0]]])

for i in range(64):
    plt.subplot(8, 8, i+1)
    act = second_conv_activation[0][0, :, :, i]
    plt.imshow(act)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))











