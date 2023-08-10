import sys
sys.path.append("../utils")
import grader

# use preloaded keras datasets and models
get_ipython().system(' mkdir -p ~/.keras/datasets')
get_ipython().system(' mkdir -p ~/.keras/models')
get_ipython().system(' ln -s $(realpath ../readonly/keras/datasets/*) ~/.keras/datasets/')
get_ipython().system(' ln -s $(realpath ../readonly/keras/models/*) ~/.keras/models/')

import numpy as np
from preprocessed_mnist import load_dataset
import keras
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
y_train,y_val,y_test = map(keras.utils.np_utils.to_categorical,[y_train,y_val,y_test])

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(X_train[0]);

import tensorflow as tf
s = tf.InteractiveSession()

import keras
from keras.models import Sequential
import keras.layers as ll

model = Sequential(name="mlp")

model.add(ll.InputLayer([28, 28]))

model.add(ll.Flatten())

# network body
model.add(ll.Dense(400))
model.add(ll.Activation('relu'))   # change activation from linear to relu

model.add(ll.Dense(200))
model.add(ll.Activation('relu'))   # change activation from linear to relu

# add 1 layer
model.add(ll.Dense(50))
model.add(ll.Activation('relu'))

# output layer: 10 neurons for each class with softmax
model.add(ll.Dense(10, activation='softmax'))

# categorical_crossentropy is your good old crossentropy
# but applied for one-hot-encoded vectors
model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

model.summary()

784 * 25 + 25

25 * 25 + 25

# fit(X,y) ships with a neat automatic logging.
#          Highly customizable under the hood.
model.fit(X_train, y_train,
          validation_data=(X_val, y_val), epochs=5);

# estimate probabilities P(y|x)
model.predict_proba(X_val[:2])

# Save trained weights
model.save("weights.h5")

print("\nLoss, Accuracy = ", model.evaluate(X_test, y_test))

# Test score...
test_predictions = model.predict_proba(X_test).argmax(axis=-1)
test_answers = y_test.argmax(axis=-1)

test_accuracy = np.mean(test_predictions==test_answers)

print("\nTest accuracy: {} %".format(test_accuracy*100))

assert test_accuracy>=0.92,"Logistic regression can do better!"
assert test_accuracy>=0.975,"Your network can do better!"
print("Great job!")

answer_submitter = grader.Grader("0ybD9ZxxEeea8A6GzH-6CA")
answer_submitter.set_answer("N56DR", test_accuracy)

answer_submitter.submit('wsdgh@qq.com', '79Ytwv7jCBjoERQf')

get_ipython().system(' rm -r /tmp/tboard/**')

from keras.callbacks import TensorBoard
model.fit(X_train, y_train, validation_data=(X_val, y_val), 
          epochs=10,
          callbacks=[TensorBoard("/tmp/tboard")])

