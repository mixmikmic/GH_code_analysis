
# This notebook modified by Adam Smith

# Original version copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0

"""An Example of a DNNClassifier for the Iris dataset."""

import argparse
import tensorflow as tf

import iris_data

# This code can be modified to read arguments from the command line, when appropriate. 
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')
args = parser.parse_args([])

# Fetch the data
(train_x, train_y), (test_x, test_y) = iris_data.load_data()

type(train_x)

train_x.head()

type(train_y)

train_y.head()

train_x.size, test_x.size


# Feature columns describe how to use the input.
# We are adding one numeric feature for each column of the training data
my_feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    

# Build 2 hidden layer DNN with 10, 10 units respectively.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
    hidden_units=[10, 10],
        # The model must choose between 3 classes.
    n_classes=3,
        ## We can also set the directory where model information will be saved.
    ##model_dir='models/iris'
    )

type(classifier)

classifier.train(
    input_fn=lambda:iris_data.train_input_fn(train_x, train_y,
                                                 args.batch_size),
    steps=args.train_steps)

# This code is copied from iris_data.py
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

classifier.get_variable_names()

# We can insect the weights and biases of the resulting model:
classifier.get_variable_value('dnn/hiddenlayer_0/kernel')

eval_result = classifier.evaluate(
        input_fn=lambda:iris_data.eval_input_fn(test_x, test_y,
                                                args.batch_size))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

# eval_result is a dictionary with a few basic statistics
for key in eval_result.keys():
    print(key, ": ", eval_result[key])



