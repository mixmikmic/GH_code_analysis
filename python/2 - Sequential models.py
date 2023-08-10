from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))

model = Sequential()
model.add(Dense(32, input_shape=(784,)))

model = Sequential()
model.add(Dense(32, batch_input_shape=(None, 784)))
# note that batch dimension is "None" here,
# so the model will be able to process batches of any size.

model = Sequential()
model.add(Dense(32, input_dim=784))

from keras.layers.recurrent import LSTM

model = Sequential()
model.add(LSTM(32, input_shape=(10, 64)))

model = Sequential()
model.add(LSTM(32, batch_input_shape=(None, 10, 64)))

model = Sequential()
model.add(LSTM(32, input_length=10, input_dim=64))

# for a multi-class classification problem
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# for a binary classification problem
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# for a mean squared error regression problem
model.compile(optimizer='rmsprop',
              loss='mse')

# for a single-input model with 2 classes (binary):

model = Sequential()
model.add(Dense(1, input_dim=784, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# generate dummy data
import numpy as np
data = np.random.random((1000, 784))
labels = np.random.randint(2, size=(1000, 1))

# train the model, iterating on the data in batches
# of 32 samples
model.fit(data, labels, nb_epoch=10, batch_size=32)

from keras.callbacks import ModelCheckpoint, EarlyStopping
save_best = ModelCheckpoint('best_model.h5', save_best_only=True)
early_stop = EarlyStopping(patience=5)

history = model.fit(..., callbacks=[save_best, early_stop])

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(100,)))
model.add(Dense(10, activation='softmax'))

print(model.to_json())

print(model.to_yaml())

from keras.models import model_from_json

model = model_from_json(json_string)
model.load_weights('my_weights_file.h5')



