import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score, classification_report

def one_hot_encoding(labels, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    one_hot = np.zeros((num_labels, num_classes))
    one_hot.flat[index_offset + labels.ravel()] = 1
    return one_hot

def next_batch(data, target, offset, 
               batch_size, train_data=True):
    """Takes the next batch from data and target
    given the offset used. Returns the data and
    target batch as well as the offset for the 
    next iteration. If the data is for training, it 
    shuffles the content."""
    start = offset
    end = offset + batch_size
    num_examples = data.shape[0]
    
    if start == 0 and train_data:
        perm = np.random.permutation(num_examples)
        data = data[perm]
        target = target[perm]
        
    if end > num_examples and train_data:
        perm = np.random.permutation(num_examples)
        data = data[perm]
        target = target[perm]
        start = 0
        end = batch_size
    elif end > num_examples:
        end = num_examples

    return end, data[start:end], target[start:end]

# Loading the data

newsgroup = np.load('./resources/newsgroup.npz')
train_data = newsgroup['train_data']
train_target = newsgroup['train_target']
test_data = newsgroup['test_data']
test_target = newsgroup['test_target']
labels = newsgroup['labels']

# Number of training instances given 
# to the network on each epoch
batch_size = 100

# Size of the input layer
input_size = train_data.shape[1]

# Number of classes (size of the output layer)
num_classes = labels.shape[0]

# Size of the hidden layers
hidden_layer_1 = 5000
hidden_layer_2 = 2000

# Define the placeholders, this are needed 
# for the network to be given data, in this case
# we have a placeholder for the data (x) and for
# the target (y). Remember as the operations
# are symbolic, we can't just feed the neural network
# the raw dataset.
x = tf.placeholder(tf.float32, [None, input_size])
y = tf.placeholder(tf.float32, [None, num_classes])

# We define a scope (important to keep named structure)
# and define the operations in the first hidden layer.
# What it basically does is take the input layer and
# apply a matrix multiplication with a non-linearity
# (the `relu` function).
with tf.name_scope('hidden_layer_1'):
    W_h1 = tf.Variable(
        tf.truncated_normal(
            [input_size, hidden_layer_1],
            stddev=1.0 / np.sqrt(input_size)
        ),
        name='W_h1'
    )
    b_h1 = tf.Variable(
        tf.zeros([hidden_layer_1]),
        name='b_h1'
    )
    h1 = tf.nn.relu(tf.matmul(x, W_h1) + b_h1)

# Same as before, we define the operations
# for the second hidden layer
with tf.name_scope('hidden_layer_2'):
    W_h2 = tf.Variable(
        tf.truncated_normal(
            [hidden_layer_1, hidden_layer_2],
            stddev=1.0 / np.sqrt(hidden_layer_1)
        ),
        name='W_h2'
    )
    b_h2 = tf.Variable(
        tf.zeros([hidden_layer_2]),
        name='b_h2'
    )
    h2 = tf.nn.relu(tf.matmul(h1, W_h2) + b_h2)

# The last layer (output), is similar to the hidden 
# layers but in this case we don't apply the 
# non-linearity as the result is needed to calculate 
# the cost via the softmax function
with tf.name_scope('output_layer'):
    W_o = tf.Variable(
        tf.truncated_normal(
            [hidden_layer_2, num_classes],
            stddev=1.0 / np.sqrt(hidden_layer_2)
        ),
        name='W_o'
    )
    b_o = tf.Variable(
        tf.zeros([num_classes]),
        name='b_o'
    )
    logits = tf.matmul(h2, W_o) + b_o

# We define the cost function as the mean of the 
# softmax cross-entropy given the labels (or target) 
# y and the result of the output layer in the
# previous step
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        labels=y, logits=logits
    )
)

# Finally, we calculate the prediction of the 
# neural network using the argmax of the logit
y_hat = tf.argmax(logits, 1)

# We define the train step that we will keep calling
# in each epoch to fit the neural network. This uses
# an optimizer algorithm (Adam in this case) to minimize
# the cost function described early.
train_step = tf.train.AdamOptimizer(0.01)    .minimize(cost)

# First we need to define a session, which is
# TensorFlow's way to excecute a piece of code.
# Then we initialize the variables given by the
# neural network code we designed previously.
# As this is a Jupyter notebook, the session is
# interactive. I recommend reading more about this
# in TensorFlow's documentation.
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# The offset is set to zero for the first time
offset = 0

# We train the network for 2000 epochs
for epoch in range(1, 2001):
    # For each epoch we obtain the batch of data
    # needed to fit the network
    offset, batch_data, batch_target =        next_batch(train_data, train_target,
                   offset, batch_size)
    # We run the train step operation (defined before)
    # and return the loss value every 100 epochs
    _, loss = sess.run(
        [train_step, cost],
        feed_dict={
            x: batch_data,
            y: one_hot_encoding(batch_target, num_classes)
        })
    if epoch % 100 == 0:
        print("Loss for epoch %02d: %.3f" % (epoch, loss))

# First we define the initial offset to zero.
# The number of test examples is needed to calculate
# the maximum number of epochs needed.
# And a list with the predictions of each batch of data
offset = 0
test_examples = test_data.shape[0]
predictions = []

# For each batch in the dataset we run the prediction
# operation (y_hat) given the data.
for _ in range(np.int(test_examples / batch_size) + 1):
    offset, batch_data, _ = next_batch(
        test_data, test_target, offset, batch_size, False)
    predictions.append(sess.run(y_hat, feed_dict={x: batch_data}))

# Finally, concatenate the predictions and check the performance
predictions = np.concatenate(predictions)
accuracy = accuracy_score(test_target, predictions)

print("Accuracy: %.2f\n" % accuracy)

print("Classification Report\n=====================")
print(classification_report(test_target, predictions))

