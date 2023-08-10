

import keras
import numpy as np
import os
import numpy as np
import keras
from keras.layers import Input, Dense, merge
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense

m = keras.models.load_model('/home/wroscoe/d2/models/all.h5')
m.summary()

[(l.name, l.trainable) for l in m.layers]

#make all layers of the loaded network trainable
for l in m.layers:
    l.trainable = False

def add_tail_network(start_tensor, end_tensor):
    #x = Flatten(name='flattened')(x)                                        # Flatten to 1D (Fully connected)
    x = Dense(100, activation='relu', name="tail_dense_1")(end_tensor)                                    # Classify the data into 100 features, make all negatives 0
    x = Dropout(.1)(x)                                                      # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Dense(50, activation='relu')(x)                                     # Classify the data into 50 features, make all negatives 0
    x = Dropout(.1)(x)                                                      # Randomly drop out 10% of the neurons (Prevent overfitting)
    #categorical output of the angle
    angle_out = Dense(15, activation='softmax', name='angle_out')(x)        # Connect every input with every output and output 15 hidden units. Use Softmax to give percentage. 15 categories and find best one based off percentage 0.0-1.0
    
    #continous output of throttle
    throttle_out = Dense(1, activation='relu', name='throttle_out')(x)      # Reduce to 1 number, Positive number only
    
    model = Model(inputs=[start_tensor], outputs=[angle_out, throttle_out])
    model.compile(optimizer='rmsprop',
                  loss={'angle_out': 'categorical_crossentropy', 
                        'throttle_out': 'mean_absolute_error'},
                  loss_weights={'angle_out': 0.9, 'throttle_out': .001})

    return model

m2 = add_tail_network(m.get_layer('img_in').input, m.get_layer('flattened').output)

#check that the layer "tail_dense_1 is in our new model.
m2.summary()

#weights of convolution layers are the same as saved model.
np.array_equal( m.layers[1].get_weights()[0], m2.layers[1].get_weights()[0])

#weights of dense layers in new model are different. 
np.array_equal( m.layers[-6].get_weights()[0], m2.layers[-6].get_weights()[0])

## convolution layers are NOT trainable
m2.layers[1].trainable

## dense layers are trainable
m2.layers[-6].trainable

template_model_path = '/home/wroscoe/models/vision_only.h5'
m2.save(template_model_path)

m3 = keras.models.load_model(v)

[(l.name, l.trainable) for l in m3.layers]

import donkeycar as dk
from donkeycar.parts.keras import KerasCategorical
from donkeycar.parts.datastore import TubGroup

cfg = dk.config.load_config('/home/wroscoe/d2/config.py')

tub_names = ','.join(['/home/wroscoe/d2/data/tub_1_17-11-18/'])


X_keys = ['cam/image_array']
y_keys = ['user/angle', 'user/throttle']

def rt(record):
    record['user/angle'] = dk.utils.linear_bin(record['user/angle'])
    return record

kl = KerasCategorical()
kl.load(template_model_path)
print('tub_names', tub_names)
if not tub_names:
    tub_names = os.path.join(cfg.DATA_PATH, '*')
tubgroup = TubGroup(tub_names)
train_gen, val_gen = tubgroup.get_train_val_gen(X_keys, y_keys, record_transform=rt,
                                                batch_size=cfg.BATCH_SIZE,
                                                train_frac=cfg.TRAIN_TEST_SPLIT)


model_name = 'vision_test1.h5'

model_path = os.path.expanduser(model_name)

total_records = len(tubgroup.df)
total_train = int(total_records * cfg.TRAIN_TEST_SPLIT)
total_val = total_records - total_train
print('train: %d, validation: %d' % (total_train, total_val))
steps_per_epoch = total_train // cfg.BATCH_SIZE
print('steps_per_epoch', steps_per_epoch)

kl.train(train_gen,
         val_gen,
         saved_model_path=model_path,
         steps=steps_per_epoch,
         train_split=cfg.TRAIN_TEST_SPLIT)



