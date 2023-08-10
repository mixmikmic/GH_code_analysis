import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import numpy

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = numpy.loadtxt("/home/teresas/notebooks/deep_learning/files/diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# We create a Sequential model and add layers one at a time until we are happy with our network topology.
model = Sequential()
# In this case, we initialize the network weights to a small random number generated from 
# a uniform distribution (‘uniform‘), in this case between 0 and 0.05 because that is the 
# default uniform weight initialization in Keras. Another traditional alternative would be 
# ‘normal’ for small random numbers generated from a Gaussian distribution.
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
# These days, better performance is achieved using the rectifier 'relu' activation function.
model.add(Dense(8, init='uniform', activation='relu'))
# We use a sigmoid on the output layer to ensure our network output is between 0 and 1 and 
# easy to map to either a probability of class 1 or snap to a hard classification of either 
# class with a default threshold of 0.5.
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
# a binary classification problem is defined in Keras as “binary_crossentropy“
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# we can see the model configuration
model.get_config()

# individual layers configuration
model.layers[2].get_config()

# counting model's parameters
model.count_params()

# (input_dim x output_dim) + output_dim
model.layers[2].count_params()

# summary and allocation of all data to show in tensorboard
callback = keras.callbacks.TensorBoard(log_dir='/home/teresas/notebooks/deep_learning/files/diabetes/',                                          histogram_freq=1, write_graph=True, write_images=False)

# Fit the model 
history = model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10, callbacks=[callback], verbose=0)

# evaluate the model: This will generate a prediction for each input and output pair 
# and collect scores, including the average loss and any metrics you have configured, such as accuracy.
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# layers fluxogram
from keras.utils.visualize_util import plot
plot(model, to_file='/home/teresas/notebooks/deep_learning/files/model.png')

from keras.models import model_from_json
import os

# serialize model to JSON
model_json = model.to_json()
with open("/home/teresas/notebooks/deep_learning/files/model_diabetes.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("/home/teresas/notebooks/deep_learning/files/model_diabetes.h5")
print("Saved model to disk")

# load json and create model
json_file = open('/home/teresas/notebooks/deep_learning/files/model_diabetes.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/home/teresas/notebooks/deep_learning/files/model_diabetes.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

