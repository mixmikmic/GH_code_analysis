#Obtain the MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#Enable interactive output for easier debugging
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
from IPython.core.debugger import set_trace
#import necessary pacakages
import tensorflow as tf
import numpy as np

mnist

#Get an idea of the expected matrix shapes from the raw input to design our tensorflow placeholders
batch_size = 20
for i in range(1):
    batch = mnist.train.next_batch(batch_size)
    print("batch[0]:{}, batch[1]:{}".format(batch[0].shape, batch[1].shape))

#First create placeholders to receive input values

#Create some test input to validate the shape
test_x_input = np.random.random((batch_size, 784))
test_y_input = np.random.random((batch_size, 10))
print("Test x input shape: {}".format(test_x_input.shape))
print("Test y input shape: {}".format(test_y_input.shape))


#Create a tensorflow placeholder of shape (batch_size, features) called x that will accept a matrix that have the shape of text_x_input, but use None
#as the batch_size so we can change that later on the fly
x = tf.placeholder(tf.float32, [None, test_x_input.shape[1]], name='x')
print("Validate shape of x: {}".format(x.get_shape().as_list()))

#Same for y, create a placeholder to hold target output classes
y = tf.placeholder(tf.float32, [None, test_y_input.shape[1]], name='y')
print("Validate shape of y: {}".format(y.get_shape().as_list()))

# x input needs to be reshaped to 4 dimensions for tf.convo2d.
# [ batch_size, height, width, depth(Channels, 1 for grayscale)]
x_4d = tf.reshape(x, [-1, 28, 28,1], name='x_4d')
print("x_4d shape: {}".format(x_4d.get_shape().as_list()))

#Create first convolutional layer with depth of 32
# ( x, kheight, kwidth, xdepth, output_depth)

#weight is a 4 dimensional filter, (height, width, input_depth, output depth)
output_depth_1 = 32

#First, understand that weights for convolutional neural networks in tensorflow are of type tf.Variable and must
#have 4 dimensions, (kernel height, kernel width, input_depth(color channels of input), output_depth)

# Create a default initial bias that is 0.1, that will be used 
# as a base for all biases in this network
initia_bias = tf.constant(0.1)
weight_1 = tf.Variable(tf.truncated_normal((5,5,1, output_depth_1),stddev=0.1) ,name='weights_1')
print("Dimensions of the first Weight/Filter: {}".format(weight_1.get_shape()))
bias_1 = tf.Variable(tf.constant(0.1, shape=([output_depth_1])), name='bias_1')
print("Bias dimensions of the first output layer will equal the number of output depth: {}".format(bias_1.get_shape()))

#Create the first convolutional layer, using x_4d as input
convo_layer1 = tf.nn.conv2d(x_4d, weight_1, strides=[1,1,1,1], padding='SAME' )
convo_layer1 = tf.nn.bias_add(convo_layer1, bias_1)
convo_layer1 = tf.nn.relu(convo_layer1) #Use relu activation function before applying max pooling

#Apply max pooling on the convolutional layer 1 . note that the standard values for kernel size and strides for max pooling 
#are ksize=[1,2,2,1] and strides=[1,2,2,1]
maxpool_layer1 = tf.nn.max_pool(convo_layer1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')




#Obtain the depth of the maxpool_layer1 in the 3rd dimension
maxpool_layer1_depth = maxpool_layer1.get_shape().as_list()[3]

#Create the 2nd convolutional layer with depth of 64 for each 5x5 patch (determind by kernel size)
convo_depth_2_output = 64

weight_2 = tf.Variable(tf.truncated_normal((5,5,maxpool_layer1_depth, convo_depth_2_output), stddev=0.1), name='weights_2')
print("Weight shapes for 2nd convolutional layer : {}".format(weight_2.get_shape().as_list()))
bias_2 = tf.Variable(tf.constant(0.1,shape=[convo_depth_2_output]), name='bias_2')


#Create the 2nd convolutional layer by applying convo2d on top of the previous maxpool_layer
convo_layer2 = tf.nn.conv2d(maxpool_layer1, weight_2, strides=[1,1,1,1], padding='SAME') + bias_2
convo_layer2 = tf.nn.relu(convo_layer2) 

#Apply the same maxpool layer
maxpool_layer2 = tf.nn.max_pool(convo_layer2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


#Create the Dense / Fully Connected layer
#First flatten the maxpool_layer2 with dimensions (None, 7, 7, 64)
mx_layer2_dims = maxpool_layer2.get_shape().as_list()
print("Preview the maxpool_layer2 output dimensions: {}".format(mx_layer2_dims))

# Now, to apply a fully connected later on this 4 dimension matrix, we'll first have to flatten back the matrix by
# reshaping it to a 2d matrix which have the shape of 7 * 7 * 64 (based on the previous maxpool output layer)
# We'll design it to automatically obtain the calculations by obtaining the shapes from the previous maxpool_layer2 and
# multiply the 1st, 2nd and 3rd dimensions together. This is useful because if we change the kernel size, we do not 
# have to manually change this value

fc_layer = tf.reshape(maxpool_layer2, [-1, mx_layer2_dims[1] * mx_layer2_dims[2] * mx_layer2_dims[3]])
print("Reshaped FC layer Shape: {}".format(fc_layer.get_shape()))

fc_layer_dims = fc_layer.get_shape().as_list()

# We'll create a first dense/hidden layer with 1024 output neurons, and a second dense layer with 
# 10 output neurons for final classification
fc_layer_1_output = 1024

weights_fc1 = tf.Variable(tf.truncated_normal([fc_layer_dims[1], 1024], stddev=0.1), name='weights_fc1')
print("Weights of Fully connected layer 1: {}".format(weights_fc1.get_shape().as_list()))
bias_fc1 = tf.Variable(tf.constant(0.1, shape=[fc_layer_1_output]), name='bias_fc1')
print("Fully connected layer 1 - Bias_fc1 shape : ", bias_fc1.get_shape().as_list())

#Perform linear regression on the dense layer, add a bias, and apply RELU activation
fc_layer1_z = tf.nn.bias_add(tf.matmul(fc_layer, weights_fc1),bias_fc1)
fc_layer_1_relu = tf.nn.relu(fc_layer1_z)
print("fc_layer1_activation shape", fc_layer_1_relu.get_shape().as_list())

#add Dropout on the first dense layer to prevent overfitting
keep_prob = tf.placeholder(tf.float32)
fc_layer_1_drop = tf.nn.dropout(fc_layer_1_relu, keep_prob)

#Final output layer
dims_fc_l1_drop = fc_layer_1_drop.get_shape().as_list()
print("Fully connected layer 1 output shape:",dims_fc_l1_drop)

# Create output with 10 classes
# Simply a linear regression on the 1st fully connected layer, and add a bias. Note: no softmax activation functions here
weights_fc2 = tf.Variable(tf.truncated_normal(shape=(dims_fc_l1_drop[1], 10), stddev=0.1), name='weights_fc2')
print("Weights shape for FC layer 2 : {}".format(weights_fc2.get_shape().as_list()))
bias_fc2 = tf.Variable(tf.constant(0.1, shape=[10]),name='bias_fc2')

fc_layer2 = tf.nn.bias_add(tf.matmul(fc_layer_1_drop, weights_fc2),bias_fc2)
print("Final FC layer2 shape, must be the same as the one hot encoded y input: {}".format(fc_layer2.get_shape().as_list()))

print("\nConvolutional Neural Network Graph creation completed!")

#Create cross_entropy to be used in Adam optimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=fc_layer2))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#Create a subset of test images
mnist_test_x = mnist.test.images[:1000]
mnist_test_x.shape
mnist_test_labels = mnist.test.labels[:1000]
mnist_test_labels.shape

print("We'll test our model on 1000 examples from the MNIST test database")

# Warning, Calling this (particularly global_variables_initializer() )  will clear the weights if it had already been trained!
# Example to obtain 5 predictions from the training dataset
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    predictions = sess.run(fc_layer2, feed_dict={x: mnist_test_x[:5], keep_prob: 1.0})
    print("fc_layer2 output shape:{}".format(predictions.shape)) #5 rows, 10 columns
    
    #Peek into 5 rows of predictions
    print("Peek into the 5 rows of final fully connected layer 2 output predictions:\n{}".format(predictions))
    
    # Calling tf.argmax with axis = 1 on a predictions, will return the index/column that contains
    # the largest value
    
    pred_argmax = sess.run(tf.argmax(predictions,1))
    print("tf argmax on predictions returns 5 values (corresponding to 5 rows) that are indexes of the largest value in each predicted row:\n{}".format(pred_argmax))
    print("\nTaking the 0th row (prediction[0]) as an example:{}\nThe higheset value in this row is:{} which is column: [{}]"
          .format(predictions[0],np.max(predictions),pred_argmax[0]))
    
    
    print("\nPeek into 5 rows of target mnist_test_labels:\n{}".format(mnist_test_labels[:5]))
    actual_argmax = sess.run(tf.argmax(mnist_test_labels[:5], 1))
    print("Index of columns with the correct target: {}".format(actual_argmax))
    
    print("Predictions vs target:\n{}\n{}".format(pred_argmax,actual_argmax))
    
    # We'll run tf.equals that will return a vector of TRUE FALSE values, if the highest prediction index in (pred_argmax)
    # equals the index in target vector, then the prediction is correct. otherwise, if the network predicted a different 
    # class which resulted in a different index in pred_argmax, then the value will be false in that vector
    correct_predictions = sess.run(tf.equal(pred_argmax,actual_argmax))
    
    print("correct_predictions vector: {}".format(correct_predictions))

    # We can obtain the accuracy of the predictions by dividing the total count of TRUE values over the entire prediction set
    # First, cast the TRUE/False values to 1 or 0 using tf.cast to turn this TRUE/FALSE observation into a mathematical problem
    prediction_numbers = sess.run(tf.cast(correct_predictions, tf.float32))
    print("correct_predictions_in_numbers: {}".format(prediction_numbers))

    # Now calculate the accuracy by simply obtaining the mean of this vector, where 1 is a correct prediction and 0 is a wrong
    # prediction
    accuracy = sess.run(tf.reduce_mean(prediction_numbers))
    print("Accuracy: {}".format(accuracy))
    print("\nNote:The accuracy at this stage should be terrible and totally random, since we have not trained the model yet.")

correct_predictions = tf.equal(tf.argmax(fc_layer2,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# Remember to create a folder called saved_models in 
# the root dir before proceeding!
# Note that models will be overwritten with the latest training, so take caution.
# modify the script here to output in a new folder if you want to retain
# some previously trained models
save_file = './saved_models/cnn_model_0.ckpt' 

saver = tf.train.Saver()

#Warning, running this cell may overwrite your previously saved model!
batch_size = 50

iterations = 20000
reporting_count = 20

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(iterations):
        batch = mnist.train.next_batch(batch_size)
        output = sess.run(train_step, feed_dict={x:batch[0], keep_prob:0.5, y:batch[1]})
#         train_step.run(feed_dict={x: batch[0], y:batch[1], keep_prob: 0.5})
        if i % int(iterations/reporting_count) == 0:
            print("Validation accuracy:{} {}% completed".format(
                sess.run(accuracy,feed_dict={x:mnist_test_x, y:mnist_test_labels, keep_prob:1.0 }), i/iterations * 100 ))
#         print("Batch {} completed!".format(i))
    print("Train Complete!")
    
    #Remember to save model!
    saver.save(sess, save_file)
    print("Model saved as {}".format(save_file))
    

print("Total number of test images: {}".format(len(mnist.test.images)))
# Create a test set of 5000 images from the whole set (only necessary to overcome RAM limitations, 
# otherwise use the full test set)
mnist_test_x_fromlast = mnist.test.images[-1000:]
mnist_test_y_fromlast = mnist.test.labels[-1000:]

best_trained_model_path = './saved_models/cnn_model_22k.ckpt' 

# Uncomment this to use your own trained model instead
save_file = best_trained_model_path

with tf.Session() as sess:
    saver.restore(sess, save_file)
    output = sess.run(accuracy, feed_dict={x:mnist_test_x_fromlast, y:mnist_test_y_fromlast, keep_prob:1.0 })
    print("Final Accuracy :{}".format(output*100))
    # Both ways are equivalent
    print("Accuracy Eval:{}".format(accuracy.eval(feed_dict={x:mnist_test_x_fromlast, y:mnist_test_y_fromlast, keep_prob:1.0 }) * 100 ))

