import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
get_ipython().magic('matplotlib inline')


mnist = input_data.read_data_sets("./mnist", one_hot=True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

print "x_train: ", x_train.shape
print "y_train: ", y_train.shape
print "x_test: ", x_test.shape
print "y_test: ", y_test.shape

def plot_mnist(data, classes, incorrect=None):
    
    for i in range(10):
        idxs = (classes == i)
        
        # for test: see last part of this ipython notbook file
        if incorrect is not None:
            idxs *= incorrect
        # get 10 images for class i
        images = data[idxs][0:10]
            
        for j in range(5):   
            plt.subplot(5, 10, i + j*10 + 1)
            plt.imshow(images[j].reshape(28, 28), cmap='gray')
            # print a title only once for each class
            if j == 0:
                plt.title(i)
            plt.axis('off')
    plt.show()

classes = np.argmax(y_train, 1)
plot_mnist(x_train, classes)

def fully_connected(x, dim_in, dim_out, name):
    with tf.variable_scope(name) as scope:
        # create variables
        w = tf.get_variable('w', shape=[dim_in, dim_out], 
                            initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
        b = tf.get_variable('b', shape=[dim_out])
        
        # create operations
        out = tf.matmul(x, w) + b
        
        return out    

# Create model
def neural_network(x, dim_in=784, dim_h=500, dim_out=10):
    # 1st hidden layer with ReLU
    h1 = fully_connected(x, dim_in, dim_h, name='h1')
    h1 = tf.nn.relu(h1)
    
    # 2nd hidden layer with ReLU
    h2 = fully_connected(h1, dim_h, dim_h, name='h2')
    h2 = tf.nn.relu(h2)
    
    # output layer with linear
    out = fully_connected(h2, dim_h, dim_out, name='out')
    
    return out

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# Construct model with default value
out = neural_network(x)

# loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out, y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss)

# Test model
pred = tf.argmax(out, 1)
target = tf.argmax(y, 1)

correct_pred = tf.equal(pred, target)
incorrect_pred = tf.not_equal(pred, target)
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

batch_size = 100

# launch the graph
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # initialize tensor variables
    tf.initialize_all_variables().run()
    # training cycle
    for epoch in range(15):
        avg_loss = 0.
        n_iters_per_epoch = int(mnist.train.num_examples / batch_size)
        # loop over all batches
        for i in range(n_iters_per_epoch):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            # run optimization op (backprop) and loss op (to get loss value)
            _, c = sess.run([optimizer, loss], feed_dict={x: x_batch, y: y_batch})
            # compute average loss
            avg_loss += c / n_iters_per_epoch
        print "Epoch %d, Loss: %.3f"% (epoch+1, avg_loss)
    print "Finished training!"
    
    print ""
    print "Test accuracy:", sess.run(accuracy, {x: mnist.test.images, y: mnist.test.labels})
    p, incorrect = sess.run([pred, incorrect_pred], {x: mnist.test.images, y: mnist.test.labels})
    
    print "Incorrect Case"
    plot_mnist(x_test, classes=p, incorrect=incorrect)

