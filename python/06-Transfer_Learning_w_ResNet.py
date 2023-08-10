#import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

get_ipython().run_line_magic('run', '__initremote__.py')

early_stop = keras.callbacks.EarlyStopping(monitor='val_acc', 
                                           min_delta=0, 
                                           patience=5, 
                                           verbose=0, 
                                           mode='auto')

resnet_base = keras.applications.resnet50.ResNet50(include_top=False, 
                                     weights='imagenet', pooling = 'avg')

resnet_base.summary()

x_train.shape

x_train[0].shape

import numpy as np
import scipy

big_x_train = np.array([scipy.misc.imresize(x_train[i], (225, 225, 3)) 
                            for i in range(0, len(x_train))]).astype('float32')

big_x_train.shape

x_preds_res_train = resnet_base.predict(big_x_train)

x_preds_res_train.shape

big_x_test = np.array([scipy.misc.imresize(x_test[i], (225, 225, 3)) 
                            for i in range(0, len(x_test))]).astype('float32')

x_preds_res_test = resnet_base.predict(big_x_test)

x_preds_res_test

x_preds_res_test.shape

import pickle
pickle.dump(x_preds_res_train, open("x_preds_res_train", "wb"))
pickle.dump(x_preds_res_test, open("x_preds_res_test", "wb"))

x_preds_res_train.shape[1:]

output = Sequential()
output.add(Dense(512, input_shape=x_preds_res_train.shape[1:]))
output.add(Activation('relu'))
output.add(Dense(10))
output.add(Activation('softmax'))

opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-5)

output.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

history_resnet = output.fit(x_preds_res_train, y_train,
              batch_size=32,
              epochs=100,
              validation_data=(x_preds_res_test, y_test),
              shuffle=True,
              callbacks=[early_stop])

