import mnist_reader
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

from keras import backend as K
img_rows = 28
img_cols = 28
if(K.image_data_format() == 'channels_first'):
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    in_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    in_shape = (img_rows, img_cols, 1)
print(X_train.shape)    
print(X_test.shape)    

import numpy as np
from keras.utils import np_utils
num_classes = 10
y_train = np_utils.to_categorical(np.array(y_train), num_classes)
y_test = np_utils.to_categorical(np.array(y_test), num_classes)
print(y_train.shape)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', 
                        input_shape=in_shape))
model.add(MaxPooling2D(pool_size=3))
model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=3))
model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=3))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])

from keras.callbacks import ModelCheckpoint  

# Save the model with best validation loss
checkpointer = ModelCheckpoint(filepath='best_weights.hdf5', 
                               verbose=1, save_best_only=True)

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=1000, validation_split=0.2, callbacks=[checkpointer], verbose=1)

model.load_weights('best_weights.hdf5')

print("\nAccuracy: ", model.evaluate(X_test, y_test)[1] * 100, "%")



