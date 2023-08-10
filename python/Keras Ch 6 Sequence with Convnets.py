from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 10000
max_len = 500

print('loading data')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print('Pad Sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
print('x_train.shape:', x_train.shape)
print('x_test.shape:', x_test.shape)

get_ipython().magic('matplotlib inline')

from keras import models
from keras import layers
from keras.optimizers import RMSprop

from util import print_curves

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


def plot_model_jupyter(model):
    SVG(model_to_dot(model).create(prog='dot', format='svg'))


model = models.Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 14, activation='relu'))
# default pool size 2
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()
plot_model_jupyter(model)

plot_model_jupyter(model)

model.compile(optimizer=RMSprop(), loss='binary_crossentropy', metrics=['acc'])
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2)

print_curves(history)

plot_model_jupyter(model)

SVG(model_to_dot(model).create(prog='dot', format='svg'))

model.evaluate(x_test, y_test)

import os
import pandas as pd

data_dir = './Downloads/'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
df = pd.read_csv(fname)

df_time = df
df = df.drop('Date Time', axis=1)
mean = df.mean(0)
std = df.std(0)
# standardize data
df = df.sub(mean, axis=1).div(std, axis=1)

import numpy as np


def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6,
              reverse=False):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback

    # create offset so we don't see the same 1/step th of data in non-shuffled
    # scenario
    offset = 1

    while True:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                # won't this reset the generator to the same place, esentially not giving us 1/6 of the data?
                # might be somewhat trivial since we're taking hourly samples,
                # so data only has changed by an hour and we're reading across
                # 8 years total dataset
                i = min_index + lookback + offset
                offset += 1
                if offset == step:
                    offset = 0
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        if reverse:
            yield samples[:, ::-1, :], targets
        else:
            yield samples, targets

lookback = 1440
step = 6
delay = 144
batch_size = 128
num_features = df.shape[1]

train_gen = generator(
    df.values,
    lookback=lookback,
    delay=delay,
    min_index=0,
    max_index=200000,
    shuffle=True,
    step=step,
    batch_size=batch_size)

val_gen = generator(
    df.values,
    lookback=lookback,
    delay=delay,
    min_index=200001,
    max_index=300000,
    step=step,
    batch_size=batch_size)

test_gen = generator(
    df.values,
    lookback=lookback,
    delay=delay,
    min_index=300001,
    max_index=None,
    step=step,
    batch_size=batch_size)

# how many train steps to see entire dataset
train_steps = (200000 - lookback) // batch_size

# how many val steps to see entire dataset
val_steps = (300000 - 200001 - lookback) // batch_size

# how many test steps to see entire dataset
test_steps = (len(df) - 300001 - lookback) // batch_size

# need to check if we're processing the cursor at all, i.e. are we only
# seeing 1/6 of the data since we're sampling every 6 hours
# Actually, we're shuffling, so that should take care of it, but only for
# training, in theory you want to run through 6 times, offestting,
# something like that

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Conv1D(32, 5, activation='relu',
                        input_shape=(None, df.shape[-1])))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 5, activation='relu'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')

history = model.fit_generator(train_gen, steps_per_epoch=500,
                              epochs=20, validation_data=val_gen,
                              validation_steps=val_steps)

print_curves(history)

lookback = 1440

# Changed step to 3 so we look at every 30 minutes
step = 3
delay = 144
batch_size = 128
num_features = df.shape[1]

train_gen = generator(
    df.values,
    lookback=lookback,
    delay=delay,
    min_index=0,
    max_index=200000,
    shuffle=True,
    step=step,
    batch_size=batch_size)

val_gen = generator(
    df.values,
    lookback=lookback,
    delay=delay,
    min_index=200001,
    max_index=300000,
    step=step,
    batch_size=batch_size)

test_gen = generator(
    df.values,
    lookback=lookback,
    delay=delay,
    min_index=300001,
    max_index=None,
    step=step,
    batch_size=batch_size)

# how many train steps to see entire dataset
train_steps = (200000 - lookback) // batch_size

# how many val steps to see entire dataset
val_steps = (300000 - 200001 - lookback) // batch_size

# how many test steps to see entire dataset
test_steps = (len(df) - 300001 - lookback) // batch_size

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Conv1D(32, 5, activation='relu',
                        input_shape=(None, df.shape[-1])))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 5, activation='relu'))
model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5))
model.add(layers.Dense(1))

model.summary()

model.compile(optimizer=RMSprop(), loss='mae')

history = model.fit_generator(train_gen, steps_per_epoch=500,
                              epochs=20, validation_data=val_gen,
                              validation_steps=val_steps)

print_curves(history)

