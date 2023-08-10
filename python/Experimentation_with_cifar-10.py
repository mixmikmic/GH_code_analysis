# packages used for machine learning
import tensorflow as tf

# packages used for processing: 
from six.moves import cPickle as pickle # for reading the data
import matplotlib.pyplot as plt # for visualization
import numpy as np
from sklearn.preprocessing import OneHotEncoder # for encoding the labels in one hot form

# for operating system related stuff
import os
import sys # for memory usage of objects
from subprocess import check_output

# to plot the images inline
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../Data/" directory.

def exec_command(cmd):
    '''
        function to execute a shell command and see it's 
        output in the python console
        @params
        cmd = the command to be executed along with the arguments
              ex: ['ls', '../input']
    '''
    print(check_output(cmd).decode("utf8"))

# check the structure of the project directory
exec_command(['ls', '../..'])

''' Set the constants for the script '''

# various paths of the files
data_path = "../../Data/cifar-10" # the data path
train_meta = os.path.join(data_path, "batches.meta")
idea = "IDEA_1"
base_model_path = '../../Models'
idea_model_path = os.path.join(base_model_path, idea)

# constant values:
size = 32 # the images of size 32 x 32
channels = 3 # RGB channels
highest_pixel_value = 255.0 # 8 bits for every channel. So, max value is 255
no_of_epochs = 200 # No. of epochs to run
no_of_batches = 5 # There are 5 batches in the dataset
checkpoint_factor = 5 # save the model after every 5 steps (epochs)
num_classes = 10 # There are 10 different classes in the dataset
k_size = 3 # all kernels are 3x3
n_hidden_neurons_in_fc_layers = 512
representation_vector_length = 128 # length of the mid_level representation vector
batch_size = 128 # we look at 64 images at a time

# check the contents inside the data folder
exec_command(['ls', data_path])

# function to unPickle a file: 
def unpickle(file):
    '''
        This function takes the file path and unPickles the file acquired from it
        @Param file: the string path of the file
        @return: The dict object unPickled from the file
    '''
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

meta_data = unpickle(train_meta)

# check it's contents
meta_data

train_batch_preliminary = unpickle(os.path.join(data_path, "data_batch_3"))

# check it's contents
train_batch_preliminary.keys()

# Extract the first 3 images from the dataset
preliminary_data = train_batch_preliminary['data'].reshape((len(train_batch_preliminary['data']), 32, 32, 3), 
                                                           order='F')
preliminary_labels = train_batch_preliminary['labels']

# view some of the data:
preliminary_data[33, :10, :10, 2] #(10 x 10) data of blue channel of 33rd image

# check a few values of the labels of the dataset
preliminary_labels[:10] 

for _ in range(3):
    random_index = np.random.randint(preliminary_data.shape[0])
    
    plt.figure().suptitle("Random Image from the dataset: %s" %(meta_data['label_names'][preliminary_labels[random_index]]))
    plt.imshow(preliminary_data[random_index], interpolation='none')

# let's try using the numpy.rot90 method for this:
random_index = np.random.randint(preliminary_data.shape[0])
    
plt.figure().suptitle("Random Image from the dataset: %s" %(meta_data['label_names'][preliminary_labels[random_index]]))
plt.imshow(np.rot90(preliminary_data[random_index], axes=(1, 0)), interpolation='none'); # suppress the unnecessary
# output

# The batch generator function:
def generateBatch(batchFile):
    '''
        The function to generate a batch of data suitable for performing the convNet operations on it
        @param batchFile -> the path of the input batchfile
        @return batch: (data, labels) -> the processed data.
    '''
    # unpickle the batch file:
    data_dict = unpickle(batchFile)
    
    # extract the data and labels from this dictionary
    unprocessed_data = data_dict['data']
    integer_labels = np.array(data_dict['labels']) # labels in integer form
    
    # reshape and rotate the data
    data = unprocessed_data.reshape((len(unprocessed_data), size, size, channels), order='F')
    processed_data = np.array(map(lambda x: np.rot90(x, axes=(1, 0)), data))
    
    # normalize the images by dividing all the pixels by 255
    processed_data = processed_data.astype(np.float32) / highest_pixel_value
    
    # encode the labels in one-hot encoded form
    # we use the sklearn.preprocessing package for doing this
    encoder = OneHotEncoder(sparse=False)
    encoded_labels = np.array(encoder.fit_transform(integer_labels.reshape(len(integer_labels), 1)))
    
    # return the processed data and the encoded_labels:
    return (processed_data, encoded_labels)

# load the batch no. 1 and check if it works correctly.
batch_data, batch_labels = generateBatch(os.path.join(data_path, "data_batch_1"))
print (batch_data.shape, batch_labels.shape)

# batch_data[0, :12, :12, 2]

# extract one image from the data and display it
randomIndex = np.random.randint(batch_data.shape[0])
randomImage = batch_data[randomIndex]
print "Random image shape: " + str(randomImage.shape)

print "Random image dataType" + str(randomImage.dtype)

print "\n\ncheck if the data has been properly normalized"
print randomImage[:3, :3, 0]

# Visualize the random image from the dataset
plt.figure()
plt.imshow(randomImage, interpolation='none'); # suppress the unnecessary

# point to reset the graph:
tf.reset_default_graph()

with tf.variable_scope("Placeholders"):
    tf_input = tf.placeholder(tf.float32, shape=(None, size, size, channels), name="inputs")
    
    # add an image summary for the tf_input
    tf_input_summary = tf.summary.image("Input_images", tf_input)
    
    tf_labels = tf.placeholder(tf.float32, shape=(None, num_classes), name="labels")
    # this is to send in the representation vector tweaked by us to generate images that we want
    tf_representation_vector = tf.placeholder(tf.float32, shape=(None, num_classes), name="representation") 

# print all these tensors to check if they have been correctly defined
tf_input, tf_labels, tf_representation_vector
# all look good

with tf.variable_scope("Weights_and_biases"):
    # special b0 for the input images to be added when performing the backward computations
    b0 = tf.get_variable("b0", shape=(1, size, size, channels), dtype=tf.float32, 
                         initializer=tf.zeros_initializer())
    
    # normal kernel weights and biases
    w1 = tf.get_variable("W1", shape=(k_size, k_size, channels, 4), dtype=tf.float32, 
                         initializer=tf.truncated_normal_initializer())
    
    b1 = tf.get_variable("b1", shape=(1, 16, 16, 4), dtype=tf.float32, 
                         initializer=tf.zeros_initializer())
    
    w2 = tf.get_variable("W2", shape=(k_size, k_size, 4, 8), dtype=tf.float32, 
                         initializer=tf.truncated_normal_initializer())
    
    b2 = tf.get_variable("b2", shape=(1, 8, 8, 8), dtype=tf.float32, 
                         initializer=tf.zeros_initializer())
    
    w3 = tf.get_variable("W3", shape=(k_size, k_size, 8, 16), dtype=tf.float32, 
                         initializer=tf.truncated_normal_initializer())
    
    b3 = tf.get_variable("b3", shape=(1, 4, 4, 16), dtype=tf.float32, 
                         initializer=tf.zeros_initializer())
    
    w4 = tf.get_variable("W4", shape=(k_size, k_size, 16, 32), dtype=tf.float32, 
                         initializer=tf.truncated_normal_initializer())
    
    b4 = tf.get_variable("b4", shape=(1, 2, 2, 32), dtype=tf.float32, 
                         initializer=tf.zeros_initializer())
    
    # two more weights and biases for the final fully connected layers
    
    w_fc1 = tf.get_variable("W_fc1", shape=(representation_vector_length, n_hidden_neurons_in_fc_layers), 
                            dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    
    b_fc1 = tf.get_variable("b_fc1", shape=(1, n_hidden_neurons_in_fc_layers), dtype=tf.float32, 
                         initializer=tf.zeros_initializer())
    
    w_fc2 = tf.get_variable("W_fc2", shape=(n_hidden_neurons_in_fc_layers, num_classes), dtype=tf.float32, 
                         initializer=tf.contrib.layers.xavier_initializer())
    
    b_fc2 = tf.get_variable("b_fc2", shape=(1, num_classes), dtype=tf.float32, 
                         initializer=tf.zeros_initializer())
    

# define a function for the forward_computations (named as encode)
def encode(inp):
    '''
        ** Note this function uses globally defined filter and bias weights
        ** activation function used is tf.abs! (AANN idea)
        Function to encode the given input images into the final num_classes-dimensional representation vector
        @param
        inp => tensor corresponding to batch of input images
        @return => tensor of shape [batch_size x num_classes] 
    '''
    stride_pattern = [1, 2, 2, 1] # define the stride pattern to halve the image everytime
    padding_pattern = "SAME" # padding pattern for the conv layers
    
    # define the convolution layers:
    z1 = tf.nn.conv2d(inp, w1, stride_pattern, padding_pattern) + b1
    a1 = tf.abs(z1)
    
    z2 = tf.nn.conv2d(a1, w2, stride_pattern, padding_pattern) + b2
    a2 = tf.abs(z2)
    
    z3 = tf.nn.conv2d(a2, w3, stride_pattern, padding_pattern) + b3
    a3 = tf.abs(z3)
    
    z4 = tf.nn.conv2d(a3, w4, stride_pattern, padding_pattern) + b4
    a4 = tf.abs(z4)
    
    # reshape the a4 activation map:
    fc_inp = tf.reshape(a4, shape=(-1, representation_vector_length))
    
    assert fc_inp.shape[-1] == representation_vector_length, "mid_level_representation_vector isn't 128 dimensional"
    
    # define the fully connected layers:
    
    z_fc1 = tf.matmul(fc_inp, w_fc1) + b_fc1
    a_fc1 = tf.abs(z_fc1)
    
    z_fc2 = tf.matmul(a_fc1, w_fc2) + b_fc2
    a_fc2 = tf.abs(z_fc2)
    
    assert a_fc2.shape[-1] == num_classes, "final_representation_vector isn't 10 dimensional"
    
    # if everything is fine, return the final activation vectors:
    return a_fc2, tf.shape(a1), tf.shape(a2), tf.shape(a3)

with tf.variable_scope("Encoder"):
    y_, sha1, sha2, sha3 = encode(tf_input)

# check the type of y_ 
print y_
# looks good alright!

def decode(inp, sha1, sha2, sha3):
    ''' 
        ** Note this function uses globally defined filter and bias weights
        ** activation function used is tf.abs! (AANN idea)
        Function to decode the given input representation vector into 
        the size - dimensional images that should be as close as possible
        @param
        inp => tensor corresponding to batch of representation vectors
        @return => tensor of shape [batch_size x size x size x channels]
    '''
    stride_pattern = [1, 2, 2, 1] # define the stride pattern to halve the image everytime
    padding_pattern = "SAME" # padding pattern for the conv layers
    
    # define the backward pass through the fully connected layers:
    z_b_1 = tf.matmul(inp, tf.transpose(w_fc2)) + b_fc1
    a_b_1 = tf.abs(z_b_1)
    
    z_b_2 = tf.matmul(a_b_1, tf.transpose(w_fc1)) + tf.reshape(b4, shape=(1, -1))
    a_b_2 = tf.abs(z_b_2)
    
    assert a_b_2.shape[-1] == representation_vector_length, "reverse_pass: vector not 128 dimensional"
    
    # reshape the vector into a feature map:
    dconv_in = tf.reshape(a_b_2, shape=(-1, 2, 2, 32)) # reshape into 2x2 maps
    
    # define the deconvolution operations
    z_b_dconv_1 = tf.nn.conv2d_transpose(dconv_in, w4, sha3, 
                                         stride_pattern, padding_pattern) + b3
    a_b_dconv_1 = tf.abs(z_b_dconv_1)

    
    z_b_dconv_2 = tf.nn.conv2d_transpose(a_b_dconv_1, w3, sha2,
                                        stride_pattern, padding_pattern) + b2
    a_b_dconv_2 = tf.abs(z_b_dconv_2)    
    
    
    z_b_dconv_3 = tf.nn.conv2d_transpose(a_b_dconv_2, w2, sha1,
                                        stride_pattern, padding_pattern) + b1
    a_b_dconv_3 = tf.abs(z_b_dconv_3)
    
    
    z_b_dconv_4 = tf.nn.conv2d_transpose(a_b_dconv_3, w1, tf.shape(tf_input),
                                        stride_pattern, padding_pattern) + b0
    a_b_dconv_4 = tf.abs(z_b_dconv_4)
    
    # return the final computed image:
    return a_b_dconv_4

with tf.variable_scope("Decoder"):
    x_ = decode(y_, sha1, sha2, sha3)
    
    # add the image summary for the x_ tensor
    x__summary = tf.summary.image("Network_generated_image", x_)

# check if the x_ is a good tensor
print x_
# looks good

# define the decoder predictions:
with tf.variable_scope("Decoder_predictions"):
    generated_image = decode(tf_representation_vector, sha1, sha2, sha3)

# check sanity of the generated_image
print generated_image
# looks good! :)

# define the predictions generated by the network in the forward direction:
def direction_cosines(vector):
    '''
        function to calculate the direction cosines of the given batch of input vectors
        @param
        vector => activations tensor 
        @return => the direction cosines of x
    '''
    sqr = tf.square(vector)
    div_val = tf.sqrt(tf.reduce_sum(sqr, axis=1, keep_dims=True))
    
    # return the direction cosines of the vector:
    return vector / div_val

# use this function to define the predictions:
with tf.variable_scope("Predictions"):
    predictions = direction_cosines(y_)

predictions

with tf.variable_scope("Forward_cost"):
    fwd_cost = tf.reduce_mean(tf.abs(predictions - tf_labels))
    
    # add scalar summary for the fwd_cost
    fwd_cost_summary = tf.summary.scalar("Forward_cost", fwd_cost)

with tf.variable_scope("Backward_cost"):
    bwd_cost = tf.reduce_mean(tf.abs(x_ - tf_input))
    
    # add a scalar summary for the bwd_cost
    bwd_cost_summary = tf.summary.scalar("Backward_cost", bwd_cost)

with tf.variable_scope("Final_cost"):
    cost = fwd_cost + bwd_cost
    
    # add a scalar summary
    cost_summary = tf.summary.scalar("Final_cost", cost)

with tf.variable_scope("Trainer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_step = optimizer.minimize(cost) # minimize the final cost

with tf.variable_scope("Errands"):
    init = tf.global_variables_initializer()
    all_summaries = tf.summary.merge_all()

model_path = os.path.join(idea_model_path, "Model_cifar_1")

''' 
    WARNING WARNING WARNING!!! This is the main training cell. Since, the data used for this task is CIFAR-10, 
    This cell will take a really really long time on low-end machines. It will however not crash your pc, since 
    I have bootstrapped the training in such a way that it loads a small chunk of data at a time to train.
    
    It took me around 5hrs to execute this cell entirely.
'''

with tf.Session() as sess:
    
    tensorboard_writer = tf.summary.FileWriter(logdir=model_path, graph=sess.graph)
    saver = tf.train.Saver(max_to_keep=2)
    
    if(os.path.isfile(os.path.join(model_path, "checkpoint"))):
        # load the weights from the model1
        # instead of global variable initializer, restore the graph:
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
    
    else:
        # initialize all the variables
        sess.run(tf.global_variables_initializer())
    
    g_step = 0
    for ep in range(205, 100 + no_of_epochs):  # epochs loop
        
        print "epoch: " + str(ep + 1)
        print "================================================================================================="
        print "================================================================================================="
        
        for batch_n in range(no_of_batches):  # batches loop
            # generate the batch images and labels
            batch_images, batch_labels = generateBatch(os.path.join(data_path, "data_batch_" + str(batch_n + 1)))
            
            min_batch_size = batch_size 
            
            print "current_batch: " + str(batch_n + 1)
            
            for index in range(int(float(len(batch_images)) / min_batch_size)):
                start = index * min_batch_size
                end = start + min_batch_size
                minX = batch_images[start: end]; minY = batch_labels[start: end]
                
                _, loss = sess.run([train_step, cost], feed_dict={tf_input: minX, tf_labels: minY})
                
                if(index % 75 == 0):
                    print('range:{} loss= {}'.format((start, end), loss))
            
                g_step += 1
                
            print "\n=========================================================================================\n"
        
        if((ep + 1) % checkpoint_factor == 0 or ep == 0):
            
            # calculate the summaries:
            sums = sess.run(all_summaries, feed_dict={tf_input: minX, tf_labels: minY})
            
            # add the summaries to the fileWriter
            tensorboard_writer.add_summary(sums, global_step = g_step)
            
            # save the model trained so far:
            saver.save(sess, os.path.join(model_path, "model_cifar_1"), global_step = (ep + 1))
        
    print "================================================================================================="
    print "================================================================================================="

with tf.Session(graph = computation_graph) as sess:
    # load the weights from the model1
    saver = tf.train.Saver()
    
    # instead of global variable initializer, restore the graph:
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    
    prediction = sess.graph.get_tensor_by_name("prediction:0")
    inputs = sess.graph.get_tensor_by_name("inputs:0")
    
    random_image = batch_data[np.random.randint(len(batch_data))]
    reconstructed_image = sess.run(prediction, feed_dict={inputs: np.array([random_image])})[0]
    
    # plot the two images with their titles:
    plt.figure().suptitle("Original Image")
    plt.imshow(random_image, interpolation='none')
    
    plt.figure().suptitle("Reconstructed Image")
    plt.imshow(reconstructed_image, interpolation='none')

