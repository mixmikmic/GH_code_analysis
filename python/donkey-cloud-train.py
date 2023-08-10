import tensorflow
print(tensorflow.__version__)

from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping

class MyEstimator():
    '''
    The Estimator creates and trains the model.
    
    Renamed from MyPilot.
    '''
    def __init__(self):
        self.model = default_categorical()

    def train(self, train_gen, val_gen, saved_model_path,
              epochs=100, steps=100, train_split=0.8, verbose=1,
              min_delta=.0005, patience=5, use_early_stop=True):

        save_best = ModelCheckpoint(saved_model_path, 
                                    monitor='val_loss', 
                                    verbose=verbose, 
                                    save_best_only=True, 
                                    mode='min')
        
        early_stop = EarlyStopping(monitor='val_loss', 
                                   min_delta=min_delta, 
                                   patience=patience, 
                                   verbose=verbose, 
                                   mode='auto')

        callbacks_list = [save_best]

        if use_early_stop:
            callbacks_list.append(early_stop)
        
        hist = self.model.fit_generator(
                        train_gen, 
                        steps_per_epoch=steps, 
                        epochs=epochs, 
                        verbose=1, 
                        validation_data=val_gen,
                        callbacks=callbacks_list, 
                        validation_steps=steps*(1.0 - train_split))
        return hist

def default_categorical(): 
    from tensorflow.python.keras.models import Model
    from tensorflow.python.keras.layers import Convolution2D
    from tensorflow.python.keras.layers import Input, Dropout, Flatten, Dense
    
    img_in = Input(shape=(120, 160, 3), name='img_in')
    x = img_in

    x = Convolution2D(24, (5,5), strides=(2,2), activation='relu')(x)
    x = Convolution2D(32, (5,5), strides=(2,2), activation='relu')(x)
    x = Convolution2D(64, (5,5), strides=(2,2), activation='relu')(x)
    x = Convolution2D(64, (3,3), strides=(2,2), activation='relu')(x)
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)

    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(.1)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(.1)(x)

    angle_out = Dense(15, activation='softmax', name='angle_out')(x)
    throttle_out = Dense(1, activation='relu', name='throttle_out')(x)
    
    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
    model.compile(optimizer='adam',
                  loss={'angle_out': 'categorical_crossentropy', 'throttle_out': 'mean_absolute_error'},
                  loss_weights={'angle_out': 0.9, 'throttle_out': .001})

    return model

# List installed python libraries. tensorflow-gpu (GPU-optimized) should be installed.
get_ipython().system('conda list | grep -i tensorflow')
get_ipython().system('conda list | grep -i donkey')

# Clone donkey git
get_ipython().run_line_magic('cd', '~/SageMaker')
get_ipython().system('rm -rf ~/SageMaker/donkey')
get_ipython().system('git clone https://github.com/wroscoe/donkey')

# Donkey has dependencies to tensorflow (non-GPU) and keras, none of which we are interested in.
# Remove Keras and replace tensorflow with tensorflow-gpu
get_ipython().system("sed -i -e '/keras==2.0.8/d' donkey/setup.py")
get_ipython().system("sed -i -e 's/tensorflow>=1.1/tensorflow-gpu>=1.4/g' donkey/setup.py")

# Install Donkey
get_ipython().system('pip uninstall donkeycar --yes')
get_ipython().system('pip install ./donkey')
get_ipython().system('pip show donkeycar')

# Define some globals for now
BATCH_SIZE = 128
TEST_SPLIT = 0.8
EPOCHS = 5               # <---- NOTE! Using only 5 epochs for now, to speed up test-training...

import os
from donkeycar.parts.datastore import TubGroup
from donkeycar.utils import linear_bin

def train(tub_names, model_name):
    '''
    Convenience method for training using MyEstimator
    
    Requires the TubGroup class from Donkey to read Tub data.
    '''
    x_keys = ['cam/image_array']
    y_keys = ['user/angle', 'user/throttle']

    def rt(record):
        record['user/angle'] = linear_bin(record['user/angle'])
        return record

    tubgroup = TubGroup(tub_names)
    train_gen, val_gen = tubgroup.get_train_val_gen(x_keys,
                                                    y_keys,
                                                    record_transform=rt,
                                                    batch_size=BATCH_SIZE,
                                                    train_frac=TEST_SPLIT)

    model_path = os.path.expanduser(model_name)

    total_records = len(tubgroup.df)
    total_train = int(total_records * TEST_SPLIT)
    total_val = total_records - total_train
    print('train: %d, validation: %d' % (total_train, total_val))
    steps_per_epoch = total_train // BATCH_SIZE
    print('steps_per_epoch', steps_per_epoch)

    kl = MyEstimator()
    kl.train(train_gen,
             val_gen,
             saved_model_path=model_path,
             steps=steps_per_epoch,
             train_split=TEST_SPLIT,
             epochs=EPOCHS)

# Download Tub
sample_data_location = 's3://jayway-robocar-raw-data/samples'
get_ipython().system('aws s3 cp {sample_data_location}/ore.zip /tmp/ore.zip')
get_ipython().system('mkdir -pv ~/SageMaker/data')
get_ipython().system('unzip /tmp/ore.zip -d ~/SageMaker/data/')

# Invoke
get_ipython().system('mkdir -pv ~/SageMaker/models')
tub = '~/SageMaker/data/tub_8_18-02-09'
model = '~/SageMaker/models/my-cloud-model'

train(tub, model)

