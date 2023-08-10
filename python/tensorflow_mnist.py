# import training data
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

save_file = 'model.ckpt'
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, save_file)

# Parameters
learning_rate = 0.02
training_epochs = 20
batch_size = 128  # Decrease batch size if you don't have enough memory
display_step = 1

n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)
n_hidden_layer = 256 # layer number of features

# Remove the previous weights and bias
tf.reset_default_graph()

# Predefine layers weight & bias
weights = {
    'hidden_layer': tf.Variable(tf.random_normal([n_input, n_hidden_layer]), name="weights_1"),
    'out': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]), name="weights_2")
}
biases = {
    'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer]), name="biases_1"),
    'out': tf.Variable(tf.random_normal([n_classes]), name="biases_2")
}

# flatten the image matrix into vectors
x = tf.placeholder("float", [None, 28, 28, 1])
y = tf.placeholder("float", [None, n_classes])

x_flat = tf.reshape(x, [-1, n_input])

# Hidden layer with RELU activation
hidden_layer = tf.add(tf.matmul(x_flat, weights['hidden_layer']), biases['hidden_layer'])
hidden_layer = tf.nn.relu(hidden_layer)
# Output layer with linear activation
output = tf.add(tf.matmul(hidden_layer, weights['out']), biases['out'])

# Define cost function to calculate the loss and an optimizer to train the network
cost = tf.reduce_mean(    tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)    .minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
session = tf.Session()
with session as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        # Display logs per epoch step
        if epoch % display_step == 0:
            c = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(c))
    saver.save(sess, save_file)
    print("Optimization Finished!")

    
    
    

with tf.Session() as sess:
    # Load the weights and bias
    saver.restore(sess, save_file)
    
    # Test the model
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # Decrease test_size if you don't have enough memory
    test_size = 256
    print("Accuracy:", accuracy.eval({x: mnist.test.images[:test_size], y: mnist.test.labels[:test_size]}))

