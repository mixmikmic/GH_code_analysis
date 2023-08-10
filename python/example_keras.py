import os.path
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, TimeDistributed, Input
from keras.layers import Conv2D, Conv3D, MaxPooling2D, LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

base_path = 'C:\\Users\\Roman Bolzern\\Desktop\\D4\\'

from notebooks.utils.keras_generator import SDOBenchmarkGenerator

# Parameters
params = {'dim': (256, 256, 4),
          'batch_size': 32,
          'shuffle': True}

# Generators
training_generator = SDOBenchmarkGenerator(os.path.join(base_path, 'train'), **params)
validation_generator = SDOBenchmarkGenerator(os.path.join(base_path, 'test'), **params)

# Design model
hmi_input = Input(shape=(256, 256, 4), dtype='uint8', name='main_input')
hmi = Conv2D(32, (3, 3), activation='relu')(hmi_input)
hmi = BatchNormalization()(hmi)
hmi = Conv2D(32, (3, 3), activation='relu')(hmi)
hmi = MaxPooling2D(pool_size=(2, 2))(hmi)
hmi = Dropout(0.25)(hmi)
hmi = BatchNormalization()(hmi)
hmi = Conv2D(64, (3, 3), activation='relu')(hmi)
hmi = BatchNormalization()(hmi)
hmi = Conv2D(64, (3, 3), activation='relu')(hmi)
hmi = MaxPooling2D(pool_size=(2, 2))(hmi)
hmi = Dropout(0.25)(hmi)
#hmi_auxiliary_output = Dense(1, activation='sigmoid', name='hmi_aux_output')(hmi)
hmi = BatchNormalization()(hmi)

date_input = Input(shape=(1,), name='date_input')
x = keras.layers.concatenate([hmi, date_input])

x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(64, activation='relu')(x)
main_output = Dense(1, activation='sigmoid', name='main_output')(x)

model = Model(inputs=[hmi_input, date_input], outputs=[main_output])

model.compile(optimizer=Adam(), loss='binary_crossentropy', loss_weights=[1.])

# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)

