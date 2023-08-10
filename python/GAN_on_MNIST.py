from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import random
get_ipython().magic('matplotlib inline')

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

train_dataset = mnist.train.images.reshape(mnist.train.images.shape[0], 28, 28) - 0.5
train_labels = mnist.train.labels
valid_dataset = mnist.validation.images.reshape(mnist.validation.images.shape[0], 28, 28) -0.5
valid_labels = mnist.validation.labels
test_dataset = mnist.test.images.reshape(mnist.test.images.shape[0], 28, 28) - 0.5
test_labels = mnist.test.labels

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def show_imagelist_as_grid(img_list, nrow, ncol):
    fig = plt.figure(figsize=(5,5))
    grid = AxesGrid(fig, 111, nrows_ncols=(nrow, ncol), axes_pad=0.05, label_mode="1")
    for i in range(nrow*ncol):
        im = grid[i].imshow(img_list[i], interpolation="none", cmap='gray', vmin=-0.5, vmax=0.5)
    plt.draw()
    plt.show()

image_size = 28
num_labels = 10
num_channels = 1 # grayscale

import numpy as np

def reformat(dataset, labels):
    dataset = dataset.reshape(
        (-1, image_size*image_size)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

dataset_mean = np.mean(train_dataset)
dataset_std = np.std(train_dataset)
print("mean and std: ", dataset_mean, dataset_std)

batch_size = 16
stddev=0.05
dropout_prob = 0.5
uni_weight = 0.0
dummy_size = 512
num_steps = 100 # changed later

# discriminator: 2 hidden layers
num_discr_layer1 = dummy_size//2
num_discr_layer2 = dummy_size//2
num_discr_layer3 = dummy_size//2

# generator: 3 hidden layers
num_gen_input_size = dummy_size
num_gen_layer1 = dummy_size*2
num_gen_layer2 = dummy_size*2
num_gen_layer3 = dummy_size*2

graph = tf.Graph()

with graph.as_default():

    # number of steps taken in training.
    global_step = tf.Variable(0)  
    
    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size*image_size))

    tf_noise = tf.placeholder(
        tf.float32, shape=(batch_size, num_gen_input_size))  

    # Variables for discriminator network.
    discr_w1 = tf.Variable(uni_weight/(image_size*image_size) + tf.truncated_normal(
        [image_size*image_size, num_discr_layer1], stddev=stddev), name='discr_w1')
    discr_b1 = tf.Variable(tf.zeros([num_discr_layer1]), name='discr_b1')
    
    discr_w2 = tf.Variable(uni_weight/num_discr_layer1 + tf.truncated_normal(
        [num_discr_layer1, num_discr_layer2], stddev=stddev), name='discr_w2')
    discr_b2 = tf.Variable(tf.zeros([num_discr_layer2]), name='discr_b2')

    discr_w3 = tf.Variable(uni_weight/num_discr_layer2 + tf.truncated_normal(
        [num_discr_layer2, num_discr_layer3], stddev=stddev), name='discr_w3')
    discr_b3 = tf.Variable(tf.zeros([num_discr_layer3]), name='discr_b3')
    
    discr_w4 = tf.Variable(uni_weight/num_discr_layer3 + tf.truncated_normal(
        [num_discr_layer3, 1], stddev=stddev), name='discr_w4')
    #discr_b3 = tf.Variable(tf.zeros([1]))

    # Variables for the generator network.
    gen_w1 = tf.Variable(uni_weight/num_gen_input_size + tf.truncated_normal(
        [num_gen_input_size, num_gen_layer1], stddev=stddev), name='gen_w1')
    gen_b1 = tf.Variable(tf.zeros([num_gen_layer1]), name='gen_b1')
    
    gen_w2 = tf.Variable(uni_weight/num_gen_layer1 + tf.truncated_normal(
        [num_gen_layer1, num_gen_layer2], stddev=stddev), name='gen_w2')
    gen_b2 = tf.Variable(tf.zeros([num_gen_layer2]), name='gen_b2')

    gen_w3 = tf.Variable(uni_weight/num_gen_layer2 + tf.truncated_normal(
        [num_gen_layer2, num_gen_layer3], stddev=stddev), name='gen_w3')
    gen_b3 = tf.Variable(tf.zeros([num_gen_layer3]), name='gen_b3')

    gen_w4 = tf.Variable(uni_weight/num_gen_layer3 + tf.truncated_normal(
        [num_gen_layer3, image_size*image_size], stddev=stddev), name='gen_w4')
    #gen_b3 = tf.Variable(tf.zeros([image_size*image_size]))

    # Model.
    def discr_model_dropout(data):
        discr_w1_do = tf.nn.dropout(discr_w1, dropout_prob)
        discr_w2_do = tf.nn.dropout(discr_w2, dropout_prob)
        discr_w3_do = tf.nn.dropout(discr_w3, dropout_prob)
        discr_w4_do = tf.nn.dropout(discr_w4, dropout_prob)
        discr_o1 = tf.nn.relu(tf.matmul(data, discr_w1_do) + discr_b1)
        discr_o2 = tf.nn.relu(tf.matmul(discr_o1, discr_w2_do) + discr_b2)
        discr_o3 = tf.nn.relu(tf.matmul(discr_o2, discr_w3_do) + discr_b3)
        discr_o4 = tf.nn.sigmoid(tf.matmul(discr_o3, discr_w4_do))
        return discr_o4

#     def discr_model(data):
#         discr_w1_do = discr_w1
#         discr_w2_do = discr_w2
#         discr_w3_do = discr_w3
#         discr_o1 = tf.nn.relu(tf.matmul(data, discr_w1_do) + discr_b1)
#         discr_o2 = tf.nn.relu(tf.matmul(discr_o1, discr_w2_do) + discr_b2)
#         discr_o3 = tf.nn.relu(tf.matmul(discr_o2, discr_w3_do) + discr_b2)
#         discr_o4 = tf.nn.sigmoid(tf.matmul(discr_o3, discr_w4_do))
#         return discr_o4
    
    # generator model, data will be noise in this case
    def gen_model(data):
        gen_w1_do = tf.nn.dropout(gen_w1, dropout_prob)
        gen_w2_do = tf.nn.dropout(gen_w2, dropout_prob)
        gen_w3_do = tf.nn.dropout(gen_w3, dropout_prob)
        gen_w4_do = tf.nn.dropout(gen_w4, dropout_prob)
        gen_o1 = tf.nn.relu(tf.matmul(data, gen_w1_do) + gen_b1)
        gen_o2 = tf.nn.relu(tf.matmul(gen_o1, gen_w2_do) + gen_b2)
        gen_o3 = tf.nn.relu(tf.matmul(gen_o2, gen_w3_do) + gen_b3)
        gen_o4 = tf.nn.tanh(tf.matmul(gen_o3, gen_w4_do))/2
        return gen_o4
        
    # computation
    discr_out_on_real = discr_model_dropout(tf_train_dataset)
    gen_out = gen_model(tf_noise)
    discr_out_on_gen = discr_model_dropout(gen_out)
                            
    # would normally be tf.reduce_mean(tf.log(1 - discr_out_on_gen))
    gen_loss = -tf.reduce_mean(tf.log(discr_out_on_gen)) 
    discr_loss = (-1)*(tf.reduce_mean(tf.log(discr_out_on_real)) + tf.reduce_mean(tf.log(1 - discr_out_on_gen)))
    
    trainable_vars = tf.trainable_variables()
    discr_vars = [ x for x in trainable_vars if 'discr_' in x.name]
    gen_vars = [ x for x in trainable_vars if 'gen_' in x.name]
        
    learn_rate = 0.01
    gen_learn_rate = tf.train.exponential_decay(learn_rate, global_step, num_steps, 0.99)
    discr_learn_rate = tf.train.exponential_decay(learn_rate, global_step, num_steps, 0.99)
    
    # Optimizers
    discr_optimizer = tf.train.GradientDescentOptimizer(discr_learn_rate).minimize(discr_loss, var_list=discr_vars)
    gen_optimizer = tf.train.GradientDescentOptimizer(gen_learn_rate).minimize(gen_loss, var_list=gen_vars)

num_steps = 10000
print_step = 100

gen_loss_total = 0.0
gen_trained = 0
discr_loss_total = 0.0
discr_trained = 0
with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (round(random.uniform(0, 100000)) + step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :]
        #show_imagelist_as_grid(batch_data.reshape(batch_size, image_size, image_size), 4,4)
        #batch_labels = train_labels[offset:(offset + batch_size), :]

        for rep in range(1):
            batch_noise = np.random.normal(loc=dataset_mean, scale=dataset_std, size=(batch_size, num_gen_input_size))
            feed_dict = {tf_noise : batch_noise}
            _, lg = session.run(
              [gen_optimizer, gen_loss], feed_dict=feed_dict)
            gen_loss_total += lg
            gen_trained += 1

        #if random.random()< abs(lg):
        #if (abs(lg)>0.5):
            #print(lg, "training the discriminator")
        
        batch_noise = np.random.normal(loc=dataset_mean, scale=dataset_std, size=(batch_size, num_gen_input_size))
        feed_dict = {tf_train_dataset : batch_data, tf_noise : batch_noise}
        _, ld = session.run(
          [discr_optimizer, discr_loss], feed_dict=feed_dict)
        discr_loss_total += ld
        discr_trained += 1
        
        if (step % print_step == print_step-1):
            if discr_trained == 0:
                discr_trained = 1
            print('Minibatch loss before step %d: discriminator %f, generator: %f' % (step+1, discr_loss_total/discr_trained, gen_loss_total/gen_trained))
            gen_loss_total = 0.0
            discr_loss_total = 0.0
            gen_trained = 0
            discr_trained = 0
            

            
    batch_noise = np.random.normal(loc=dataset_mean, scale=dataset_std, size=(batch_size, num_gen_input_size))
    feed_dict = {tf_noise : batch_noise}
    example_outs = gen_out.eval(feed_dict=feed_dict)
    img_list = example_outs.reshape(batch_size, image_size, image_size)
    show_imagelist_as_grid(img_list, 4, 4)

def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))

