from tensorflow.python import keras
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

keras.__version__

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)  # one-hot
test_labels = to_categorical(test_labels)

def data_generator(x, y, batch_size=32):
    batches = int(len(x)/batch_size)
    while 1:
        for i in range(batches):
            yield x[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size]

img_size=(28, 28, 1)
class NeuralNet:
    def __init__(self, use_batch_norm, activation='relu'):
        self.use_batch_norm = use_batch_norm
        self.build_model(activation=activation)
    def add_dense_layer(self, units, activation='relu'):
        if self.use_batch_norm:
            self.model.add(layers.Dense(units, use_bias=False))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.Activation(activation))
        else:
            self.model.add(layers.Dense(units, activation=activation))
    def add_conv2d_layer(self, filters, kernel_size, activation='relu', **kwargs):
        if self.use_batch_norm:
            self.model.add(layers.Conv2D(filters, kernel_size, use_bias=False, **kwargs))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.Activation(activation))
        else:
            self.model.add(layers.Conv2D(filters, kernel_size, activation=activation, **kwargs))
    def build_model(self, activation):
        self.model = models.Sequential()
        self.add_conv2d_layer(32, (3, 3), activation=activation, input_shape=img_size)
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.add_conv2d_layer(64, (3, 3), activation=activation)
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.add_conv2d_layer(64, (3, 3), activation=activation)
        self.model.add(layers.Flatten())
        self.add_dense_layer(64, activation=activation)
        self.add_dense_layer(10, activation='softmax')
    def train(self, learning_rate=0.001, epoches=40, batch_size=32, steps_per_epoch=30):
        self.model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              metrics=['accuracy'])
        history = self.model.fit_generator(generator=data_generator(train_images, train_labels, batch_size=batch_size),
                                      steps_per_epoch = steps_per_epoch,
                                      epochs = epoches,
                                      validation_data=data_generator(test_images, test_labels, batch_size=batch_size),
                                      validation_steps=len(test_images)/batch_size)
        return history.history

def plot_training_accuracies(*args, **kwargs):
    """
    Displays a plot of the accuracies calculated during training to demonstrate
    how many iterations it took for the model(s) to converge.
    
    :param args: One or more NeuralNet objects
        You can supply any number of NeuralNet objects as unnamed arguments 
        and this will display their training accuracies. Be sure to call `train` 
        the NeuralNets before calling this function.
    :param kwargs: 
        You can supply any named parameters here, but `steps_per_epoch` is the only
        one we look for. It should match the `steps_per_epoch` value you passed
        to the `train` function.
    """
    fig, ax = plt.subplots()

    steps_per_epoch = kwargs['steps_per_epoch']
    
    for history in args:
        ax.plot(range(0,len(history['acc'])*steps_per_epoch,steps_per_epoch),
                history['acc'], label="acc-" + history['name'])
        ax.plot(range(0,len(history['val_acc'])*steps_per_epoch,steps_per_epoch),
                history['val_acc'], label="val_acc-"+history['name'])
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy During Training')
    ax.legend(loc=4)
    #ax.set_ylim([0,1])
    #plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(True)
    plt.show()

    
def train_and_test(learning_rate=0.001, activation="relu", epochs=40, steps_per_epoch=30):
    nn = NeuralNet(use_batch_norm=False, activation=activation)
    bn = NeuralNet(use_batch_norm=True, activation=activation)
    history_nn = nn.train(learning_rate=learning_rate, epoches=epochs, steps_per_epoch=steps_per_epoch)
    history_bn = bn.train(learning_rate=learning_rate, epoches=epochs, steps_per_epoch=steps_per_epoch)
    history_nn['name'] = "Without batch normalization"
    history_bn['name'] = "With batch normalization"
    plot_training_accuracies(history_nn, history_bn, steps_per_epoch=steps_per_epoch)

train_and_test(learning_rate=0.001, activation='relu', epochs=3, steps_per_epoch=1875)

train_and_test(learning_rate=0.001, activation='sigmoid', epochs=3, steps_per_epoch=1875)

train_and_test(learning_rate=0.01, activation='relu', epochs=3, steps_per_epoch=1875)

train_and_test(learning_rate=0.01, activation='sigmoid', epochs=3, steps_per_epoch=1875)

train_and_test(learning_rate=0.05, activation='relu', epochs=3, steps_per_epoch=1875)

train_and_test(learning_rate=0.05, activation='sigmoid', epochs=3, steps_per_epoch=1875)

train_and_test(learning_rate=0.001, activation='relu', epochs=10, steps_per_epoch=int(1875/10))



