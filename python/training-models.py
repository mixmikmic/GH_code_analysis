import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.__version__

feature_columns = [
    tf.feature_column.numeric_column(
    "pixels", shape=784)
]

logdir = './model/'

get_ipython().system('bash ./download.sh')

classifier = tf.estimator.LinearClassifier(
    feature_columns=feature_columns, 
    n_classes=10,
    model_dir=logdir + 'linear')

def make_input_fn(data, batch_size, num_epochs, shuffle):
    return tf.estimator.inputs.numpy_input_fn(
             x={'pixels': data.images},
             y=data.labels.astype(np.int64),
             batch_size=batch_size,
             num_epochs=num_epochs,
             shuffle=shuffle)

DATA_SETS = input_data.read_data_sets("./data")

classifier.train(input_fn=make_input_fn(DATA_SETS.train, 
                               batch_size=100, 
                               num_epochs=2, 
                               shuffle=True))

accuracy_score = classifier.evaluate(
    input_fn=make_input_fn(
        DATA_SETS.test, 
        batch_size=100, 
        num_epochs=1, 
        shuffle=False))['accuracy']

accuracy_score

## Add your code here

deep_classifier.train(input_fn=make_input_fn(DATA_SETS.train, 
                               batch_size=100, 
                               num_epochs=2, 
                               shuffle=True))

accuracy_score = deep_classifier.evaluate(
    input_fn=make_input_fn(
        DATA_SETS.test, 
        batch_size=100, 
        num_epochs=1, 
        shuffle=False))['accuracy']

accuracy_score

## Your code goes here

deep_classifier.train(input_fn=make_input_fn(DATA_SETS.train, 
                               batch_size=100, 
                               num_epochs=2, 
                               shuffle=True))

