from keras import backend as K
from keras.datasets import mnist
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Conv2D, Activation, Dense, Dropout, Lambda
from keras.layers import BatchNormalization, MaxPooling2D, Flatten, Conv1D
from keras.utils import to_categorical
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from keras import optimizers
from keras import regularizers
from keras import losses
import numpy as np
from scipy.io import matlab
import os

img_rows, img_cols = 40, 40
num_classes = 10

(t_x_train, t_y_train), _ = mnist.load_data()

reps = 1

t_x_train = np.repeat(t_x_train, reps, axis=0)
x_train = np.zeros((t_x_train.shape[0], 40, 40), dtype=np.float32)

for i in range(0, x_train.shape[0]):
    x, y = np.random.randint(0, 12, 2)
    x_train[i, y:y+28, x:x+28] = t_x_train[i]

y_train = np.repeat(t_y_train, reps, axis=0)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_train /= 255.0

y_train = to_categorical(y_train, num_classes)

input_shape = (img_rows, img_cols, 1)

l2 = regularizers.l2(l=0.001)

inp = Input(shape=input_shape)
l = inp

l = Conv2D(8, (3, 3), activation='relu', kernel_regularizer=l2)(l)
l = BatchNormalization()(l)
l = MaxPooling2D((2, 2))(l)
l = Conv2D(16, (3, 3), activation='relu', kernel_regularizer=l2)(l) 
l = BatchNormalization()(l)
l = MaxPooling2D((2, 2))(l)
l = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2)(l)
l = BatchNormalization()(l)
l = MaxPooling2D((2, 2))(l)
l = Flatten()(l)
l = Dense(72, activation='relu', kernel_regularizer=l2)(l)
l = BatchNormalization()(l)
l = Dense(10, activation='softmax', kernel_regularizer=l2)(l)

model = Model(inputs=inp, outputs=l, name='40x40_input_cnn')
model.summary()

def caps_objective(y_true, y_pred):
    return K.sum(y_true * K.clip(0.8 - y_pred, 0, 1) ** 2 + 0.5 * (1 - y_true) * K.clip(y_pred - 0.3, 0, 1) ** 2)

optimizer = optimizers.Adam(lr=0.001)
model.compile(loss=losses.categorical_crossentropy, #caps_objective,
              optimizer=optimizer,
              metrics=['accuracy'])

w_path = os.path.join('weights', model.name)

if not os.path.exists(w_path):
    os.makedirs(w_path)
    
f_name = os.path.join(w_path, 'weights.{epoch:02d}.hdf5')    

# model.load_weights('weights/40x40_input_cnn/weights.03.hdf5')

batch_size = 32
num_epochs = 32

h = model.fit(x_train, y_train,
          batch_size=batch_size, epochs=num_epochs, initial_epoch=0,
          verbose=1, validation_split=0.09,
          callbacks=[ModelCheckpoint(f_name)])

f = matlab.loadmat('affnist/test.mat')

x_test = f['affNISTdata'][0, 0][2].astype(np.float32).transpose()
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
x_test /= 255
input_shape = (img_rows, img_cols, 1)

y_test = f['affNISTdata'][0, 0][4].transpose()

score = model.evaluate(x_test, y_test, verbose=1)
print('')
print('Test score: ', score[0])
print('Test accuracy: ', score[1])    



