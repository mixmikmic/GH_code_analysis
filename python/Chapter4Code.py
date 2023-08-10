import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Layer Sizes, Input Size, and the Size of the total number of classes
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)


# Network Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Create the Placeholder Variables
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

weights = {
 'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])), #784x256
 'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])), #256x256
 'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes])) #256x10
}
biases = {
 'b1': tf.Variable(tf.random_normal([n_hidden_1])), #256x1
 'b2': tf.Variable(tf.random_normal([n_hidden_2])), #256x1
 'out': tf.Variable(tf.random_normal([n_classes])) #10x1
}

def feedforward_network(x, weights, biases):
    ## First layer; a hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])    
    layer_1 = tf.nn.relu(layer_1)

    # Second layer; a hidden layer with RELU activation function
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']),biases['b2']) 
    layer_2 = tf.nn.relu(layer_2)


    # Output layer; utilizes a linear activation function
    outputLayer = tf.matmul(layer_2, weights['out']) + biases['out'] 
    
    ## Reutrn the Last Layer
    return outputLayer

# Construct model
pred = feedforward_network(x, weights, biases)

# Define the optimizer and the loss function for the network 
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initialize the Tensorflow Variables
init = tf.global_variables_initializer()

## Run the Traininng Process Using a Tensorflow Session
with tf.Session() as sess:
    sess.run(init)

    # We'll run the training cycle for the amount of epochs that we defined above
    for epoch in range(training_epochs):
        avg_loss = 0.  # Initialize the loss at zero
        total_batch = int(mnist.train.num_examples/batch_size)
        
        # Now, loop over all of the batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            
            # Here, we'll run the session by feeding in the optimizer, loss operation, and the batches of data
            _, loss = sess.run([train_op, loss_op], feed_dict={x: batch_x, y: batch_y})
            
            # Compute average loss
            avg_loss += loss / total_batch
            
        # Print out the loss at each step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "loss={:.9f}".format(avg_loss))
            
        # Test the Model's Accuracy
        pred = tf.nn.softmax(pred)  
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        
        # Calculate the accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

## Cross Entropy Loss from Scratch
def CrossEntropy(yHat, y):
    if yHat == 1:
        return -log(y)
    else:
        return -log(1 - y)

## Vanilla Gradient Descent from Scratch
def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.size  
    for i in range(num_iters):
        y_hat = np.dot(X, theta)
        theta = theta - alpha * (1.0/m) * np.dot(X.T, y_hat-y)
    return theta

## Stochastic Gradient Descent from Scratch
def SGD(f, theta0, alpha, num_iters):
    start_iter = 0
    theta= theta0
    for iter in xrange(start_iter + 1, num_iters + 1):
        _, grad = f(theta)
        theta = theta - (alpha * grad) 
    return theta

## Parametric ReLu
def parametric_relu(_x):
 alphas = tf.get_variable('alpha', _x.get_shape()[-1],
 initializer=tf.constant_initializer(0.0),
 dtype=tf.float32)
 pos = tf.nn.relu(_x)
 neg = alphas * (_x - abs(_x)) * 0.5
 return pos + neg

