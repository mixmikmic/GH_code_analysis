import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time
from tqdm import tqdm

mnist_data = input_data.read_data_sets("MNIST_dataset", one_hot=True)

def weights_init(shape):
    '''
    This function is used when weights are initialized.
    
    Input: shape - list of int numbers which are representing dimensions of our weights.
    '''
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def bias_init(shape):
    '''
    This function is used when biases are initialized.
    
    Input: shape - scalar that represents length of bias vector for particular layer in a network.
    '''
    return tf.Variable(tf.constant(0.05, shape=shape))

def conv2d_custom(input, filter_size, number_of_channels, number_of_filters, strides=(1, 1), padding='SAME', 
                  activation=tf.nn.relu, max_pool=True):
    
    '''
    This function is used to create single convolution layer in a CNN network.
    
    Inputs: input
            filter_size - int value that represents width and height for kernel used in this layer.
            number_of_channels - number of channels that INPUT to this layer has.
            number_of_filters - how many filters in our output do we want, this is going to be number of channels of this layer
                                and this number is used as a number of channels for the next layer.
            strides - how many pixels filter/kernel is going to move per time.
            paddign - if its needed we pad image with zeros. "SAME" = output has same dimensions as an input, "VALID" - this is
                      another option for padding parameter.
            activation - which activation/if any this layer will use
            max_pool - if True output height and width will be half sized the input size.  
    '''
    
    weights = weights_init([filter_size, filter_size, number_of_channels, number_of_filters])
    biases = bias_init([number_of_filters])
    
    layer = tf.nn.conv2d(input, filter=weights, strides=[1, strides[0], strides[1], 1], padding=padding) + biases
    layer = activation(layer) 
    
    if max_pool:
        layer = tf.nn.max_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    return layer 

def flatten(layer):
    '''
    This function should be used AFTER last conv layer in a network.
    
    This function will take LAYER as an input and output flattend layer. This should be done so we can use fc layer afterwards. 
    '''
    shape = layer.get_shape()
    num_of_elements = shape[1:4].num_elements()
    reshaped = tf.reshape(layer, [-1, num_of_elements])
    return reshaped, num_of_elements 

def fully_connected_layer(input, input_shape, output_shape, activation=tf.nn.relu, dropout=None):
    '''
    This function is used to create single fully connected layer in a network.
    
    Inputs: input
            intput_shape - number of "neurons" of the input to this layer
            output_shape - number of "neurons" that we want to have in this layer
            activation - which activation/if any this layer will use
            dropout - if this is NOT None but some number, we are going to, randomly, turn off neurons in this layer.
    '''
    
    weights = weights_init([input_shape, output_shape])
    biases = bias_init([output_shape])
    
    layer = tf.matmul(input, weights) + biases
    
    if activation != None:
        layer = activation(layer)
        
    if dropout != None:
        layer = tf.nn.dropout(layer, dropout)
        
    return layer

#Creating inputs to our network graph
inputs = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="Inputs")
targets = tf.placeholder(tf.float32, shape=[None, 10], name="Targets")
y_true = tf.argmax(targets, 1)

#This is where it comes together by using all of our helper functions
conv_1 = conv2d_custom(inputs, 5, 1, 16)
conv_2 = conv2d_custom(conv_1, 5, 16, 32)
conv_3 = conv2d_custom(conv_2, 5, 32, 64)
flat_layer, num_elements = flatten(conv_3)
fc_1 = fully_connected_layer(flat_layer, num_elements, 128)
logits = fully_connected_layer(fc_1, 128, 10, activation=None)

#For testing
predictions = tf.nn.softmax(logits)

y_pred_cls = tf.argmax(predictions, 1)

#Calculating cross entropy loss function and we are using Adam optimizer to optimize our network over time.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets))
optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)

#These two lines are used to get accuracy for our model
correct_prediction = tf.equal(y_pred_cls, y_true)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()

session.run(tf.global_variables_initializer())

batch_size = 32

total_number_trained = 0
epochs = 30
def optmizer():

    for i in (range(epochs)):
        epoch_loss = []
        for ii in range(mnist_data.train.num_examples//batch_size):
            batch = mnist_data.train.next_batch(batch_size)
            imgs = batch[0].reshape((-1, 28, 28, 1))
            labs = batch[1]

            dict_input = {inputs:imgs, targets:labs}

            c, _ = session.run([cost, optimizer], feed_dict=dict_input)
            epoch_loss.append(c)
        print("Epoche: {}/{}".format(i, epochs), "| Training accuracy: ", session.run(accuracy, feed_dict=dict_input), 
              "| Cost: {}".format(np.mean(epoch_loss)))

def validation_test_model():
    return session.run(accuracy, feed_dict={inputs: mnist_data.validation.images.reshape((-1, 28, 28, 1)), 
                                targets: mnist_data.validation.labels})

def test_model():
    return session.run(accuracy, feed_dict={inputs: mnist_data.test.images.reshape((-1, 28, 28, 1)), 
                                targets: mnist_data.test.labels})

optmizer()

print("Accuracy on the validation set {}".format(validation_test_model()))

print("Accuracy on the test set {}".format(test_model()))

