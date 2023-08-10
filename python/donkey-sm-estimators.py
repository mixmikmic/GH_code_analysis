import os
import glob
import pandas as pd
from PIL import Image

def read_tub(path):
    '''
    Read a Tub directory into memory
    
    A Tub contains records in json format, one file for each sample. With a default sample frequency of 20 Hz,
    a 5 minute drive session will contain roughly 6000 files.
    
    A record JSON object has the following properties (per default):
    - 'user/angle'      - wheel angle
    - 'user/throttle'   - speed
    - 'user/mode'       - drive mode (.e.g user or pilot)
    - 'cam/image_array' - relative path to image
    
    Returns a list of dicts, [ { 'record_id', 'angle', 'throttle', 'image', } ]
    '''

    def as_record(file):
        '''Parse a json file into a Pandas Series (vector) object'''
        return pd.read_json(file, typ='series')
    
    def is_valid(record):
        '''Only records with angle, throttle and image are valid'''
        return hasattr(record, 'user/angle') and hasattr(record, 'user/throttle') and hasattr(record, 'cam/image_array')
        
    def map_record(file, record):
        '''Map a Tub record to a dict'''
        # Force library to eager load the image and close the file pointer to prevent 'too many open files' error
        img = Image.open(os.path.join(path, record['cam/image_array']))
        img.load()
        # Strip directory and 'record_' from file name, and parse it to integer to get a good id
        record_id = int(os.path.splitext(os.path.basename(file))[0][len('record_'):])
        return {
            'record_id': record_id,
            'angle': record['user/angle'],
            'throttle': record['user/throttle'],
            'image': img
        }
    
    json_files = glob.glob(os.path.join(path, '*.json'))
    records = ((file, as_record(file)) for file in json_files)
    return list(map_record(file, record) for (file, record) in records if is_valid(record))

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Convolution2D
from tensorflow.python.keras.layers import Input, Dropout, Flatten, Dense
from tensorflow.python.keras.estimator import model_to_estimator

INPUT_TENSOR_NAME = "images"
SIGNATURE_NAME = "serving_default"
LEARNING_RATE = 0.001


def model_fn(features, labels, mode, params):
    '''
    Model function for Estimator.
    '''
    
    # 1. Create the model as before, but use features array as input
    x = features[INPUT_TENSOR_NAME]
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
    
    # 2. Create the TensorFlow Estimator from the Keras model (Dunno if this gonna work...)
    return model_to_estimator(keras_model=model)


def train_input_fn(training_dir, params):
    # TODO: Need to read a Tub into correct format. Could use read_tub from intro, or Donkey library TubGroup???
    records = read_tub(training_dir)
    df = pd.DataFrame.from_records(records).set_index('record_id')
    
    # TODO: Where do we define the input format (nbr of channels):
    # img_in = Input(shape=(120, 160, 3), name='img_in')
    
    # Return numpy_input_fn
    return tf.estimator.inputs.numpy_input_fn(
        x={INPUT_TENSOR_NAME: np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True)()

def eval_input_fn(training_dir, params):
    # TODO: Copy/paste train_input_fn
    return tf.estimator.inputs.numpy_input_fn(
        x={INPUT_TENSOR_NAME: np.array(training_set.data)},
        y=np.array(training_set.target),
        shuffle=True)()

def serving_input_fn(params):
    # TODO: Only finish if actually needed. We do not host the endpoint on SageMaker, only train the model
    return None



