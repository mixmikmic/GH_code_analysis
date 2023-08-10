import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

# split the training and testing data into X (image) and Y (label) arrays
train_df = pd.read_csv(r'data\fashion-mnist_train.csv')
test_df = pd.read_csv(r'data\fashion-mnist_test.csv')

train_data = np.array(train_df, dtype='float32')
test_data = np.array(test_df, dtype='float32')

x_train = train_data[:, 1:] / 255
y_train = train_data[:, 0]

x_test = test_data[:, 1:] / 255
y_test = test_data[:, 0]

x_train, x_validate, y_train, y_validate = train_test_split(
    x_train, y_train, test_size=0.2, random_state=12345,
)

im_rows = 28
im_cols = 28
batch_size = 512
im_shape = (im_rows, im_cols, 1)

x_test = x_test.reshape(x_test.shape[0], *im_shape)
x_train = x_train.reshape(x_train.shape[0], *im_shape)
x_validate = x_validate.reshape(x_validate.shape[0], *im_shape)

cnn_model_40 = Sequential([
    Conv2D(32, kernel_size=3, activation='relu', input_shape=im_shape, name='Conv2D-1'),
    MaxPooling2D(pool_size=2, name='MaxPool'),
    Dropout(0.4, name='Dropout'),
    Flatten(name='flatten'),
    Dense(32, activation='relu', name='Dense'),
    Dense(10, activation='softmax', name='Output')
], name='Dropout_40')

cnn_model_20 = Sequential([
    Conv2D(32, kernel_size=3, activation='relu', input_shape=im_shape, name='Conv2D-1'),
    MaxPooling2D(pool_size=2, name='MaxPool'),
    Dropout(0.2, name='Dropout'),
    Flatten(name='flatten'),
    Dense(32, activation='relu', name='Dense'),
    Dense(10, activation='softmax', name='Output')
], name='Dropout_20')

cnn_model_00 = Sequential([
    Conv2D(32, kernel_size=3, activation='relu', input_shape=im_shape, name='Conv2D-1'),
    MaxPooling2D(pool_size=2, name='MaxPool'),
    Flatten(name='flatten'),
    Dense(32, activation='relu', name='Dense'),
    Dense(10, activation='softmax', name='Output')
], name='Dropout_00')

cnn_models = [cnn_model_40, cnn_model_20, cnn_model_00]

cnn_model_40.summary()

cnn_model_20.summary()

cnn_model_00.summary()

# train the models and save results to a dict

history_dict = {}

for model in cnn_models:
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy']
    )
    
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=50, verbose=1,
        validation_data=(x_validate, y_validate)
    )
    
    history_dict[model.name] = history

plt.rcParams.update({'font.size': 15})
def plot_results():
    for history in history_dict:
        val_acc = history_dict[history].history['val_acc']
        val_loss = history_dict[history].history['val_loss']
        ax1.plot(val_acc, label=history, lw=2)
        ax2.plot(val_loss, label=history, lw=2)

    ax1.set_ylabel('validation accuracy', fontsize=17)
    ax2.set_ylabel('validation loss', fontsize=17)
    ax2.set_xlabel('epochs')
    ax1.legend()
    ax2.legend()
    plt.show()

fig, (ax1, ax2) = plt.subplots(2, figsize=(14, 9))
plot_results()



