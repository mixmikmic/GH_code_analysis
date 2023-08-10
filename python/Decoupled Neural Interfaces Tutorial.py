import random
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm # Used to display training progress bar

sg_sess = tf.Session()
backprop_sess = tf.Session()

# Data scaled to [0,1] interval, and labels in one-hot format
# 55k train, 5k validation, 10k test
MNIST = input_data.read_data_sets("data/", one_hot=True)

# Hyperparameters
iterations = 500000
batch_size = 250 # modified to evenly divide dataset size

init_lr = 3e-5
lr_div = 10
lr_div_steps = set([300000, 400000])

update_prob = 0.2 # Probability of updating a decoupled layer

validation_checkpoint = 10 # How often (iterations) to validate model

# Functions for constructing layers
def dense_layer(inputs, units, name, output=False):
    with tf.variable_scope(name):
        x = tf.layers.dense(inputs, units, name="fc")
        if not output:
            x = tf.layers.batch_normalization(x, name="bn")
            x = tf.nn.relu(x, name="relu")
    return x

def sg_module(inputs, units, name, label):
    with tf.variable_scope(name):
        inputs_c = tf.concat([inputs, label], 1)
        x = tf.layers.dense(inputs_c, units, name="fc", kernel_initializer=tf.zeros_initializer())
    return x

# Ops for network architecture
with tf.variable_scope("architecture"):
    # Inputs
    with tf.variable_scope("input"):
        X = tf.placeholder(tf.float32, shape=(None, 784), name="data") # Input
        Y = tf.placeholder(tf.float32, shape=(None, 10), name="labels") # Target
    
    # Inference layers
    h1 = dense_layer(X, 256, "layer1")
    h2 = dense_layer(h1, 256, name="layer2")
    h3 = dense_layer(h2, 256, name="layer3")
    logits = dense_layer(h3, 10, name="layer4", output=True)
    
    # Synthetic Gradient layers
    d1_hat = sg_module(h1, 256, "sg2", Y)
    d2_hat = sg_module(h2, 256, "sg3", Y)
    d3_hat = sg_module(h3, 256, "sg4", Y)

# Collections of trainable variables in each block
layer_vars = [tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="architecture/layer1/"),
              tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="architecture/layer2/"),
              tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="architecture/layer3/"),
              tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="architecture/layer4/")]
sg_vars = [None,
           tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="architecture/sg2/"),
           tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="architecture/sg3/"),
           tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="architecture/sg4/")]

# Function for optimizing a layer and its synthetic gradient module
def train_layer_n(n, h_m, h_n, d_hat_m, class_loss, d_n=None):
    with tf.variable_scope("layer"+str(n)):
        layer_grads = tf.gradients(h_n, [h_m]+layer_vars[n-1], d_n)
        layer_gv = list(zip(layer_grads[1:],layer_vars[n-1]))
        layer_opt = tf.train.AdamOptimizer(learning_rate=learning_rate).apply_gradients(layer_gv)
    with tf.variable_scope("sg"+str(n)):
        d_m = layer_grads[0]
        sg_loss = tf.divide(tf.losses.mean_squared_error(d_hat_m, d_m), class_loss)
        sg_opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(sg_loss, var_list=sg_vars[n-1])
    return layer_opt, sg_opt

# Ops for training
with tf.variable_scope("train"):
    with tf.variable_scope("learning_rate"):
        learning_rate = tf.Variable(init_lr, dtype=tf.float32, name="lr")
        reduce_lr = tf.assign(learning_rate, learning_rate/lr_div, name="lr_decrease")

    pred_loss = tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=logits, scope="prediction_loss")
    
    # Optimizers when using synthetic gradients
    with tf.variable_scope("synthetic"):
        layer4_opt, sg4_opt = train_layer_n(4, h3, pred_loss, d3_hat, pred_loss)
        layer3_opt, sg3_opt = train_layer_n(3, h2, h3, d2_hat, pred_loss, d3_hat)
        layer2_opt, sg2_opt = train_layer_n(2, h1, h2, d1_hat, pred_loss, d2_hat)
        with tf.variable_scope("layer1"):
            layer1_opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(h1, var_list=layer_vars[0], grad_loss=d1_hat)
    
    # Optimizer when using backprop
    with tf.variable_scope("backprop"):
        backprop_opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(pred_loss)
        
init = tf.global_variables_initializer()

# Ops for validation and testing (computing classification accuracy)
with tf.variable_scope("test"):
    preds = tf.nn.softmax(logits, name="predictions")
    correct_preds = tf.equal(tf.argmax(preds,1), tf.argmax(Y,1), name="correct_predictions")
    accuracy = tf.reduce_sum(tf.cast(correct_preds,tf.float32), name="correct_prediction_count") / batch_size

# Ops for tensorboard summary data
with tf.variable_scope("summary"):
    cost_summary_opt = tf.summary.scalar("loss", pred_loss)
    accuracy_summary_opt = tf.summary.scalar("accuracy", accuracy)
    summary_op = tf.summary.merge_all()

# Train using backprop as benchmark
with backprop_sess.as_default():
    backprop_train_writer = tf.summary.FileWriter("logging/backprop/train")
    backprop_validation_writer = tf.summary.FileWriter("logging/backprop/validation")

    backprop_sess.run(init)
    for i in tqdm(range(1,iterations+1)):
        if i in lr_div_steps: # Decrease learning rate
            backprop_sess.run(reduce_lr)
        
        data, target = MNIST.train.next_batch(batch_size)
        _, summary = backprop_sess.run([backprop_opt, summary_op], feed_dict={X:data,Y:target})
        backprop_train_writer.add_summary(summary, i)
        
        if i % validation_checkpoint == 0:
            Xb, Yb = MNIST.test.next_batch(batch_size)
            summary = backprop_sess.run([summary_op], feed_dict={X:Xb,Y:Yb})[0]
            backprop_validation_writer.add_summary(summary, i)

    # Cleanup summary writers
    backprop_train_writer.close()
    backprop_validation_writer.close()

# Train using synthetic gradients
with sg_sess.as_default():
    sg_train_writer = tf.summary.FileWriter("logging/sg/train", sg_sess.graph)
    sg_validation_writer = tf.summary.FileWriter("logging/sg/validation")
    
    sg_sess.run(init)
    for i in tqdm(range(1,iterations+1)):
        if i in lr_div_steps: # Decrease learning rate
            sg_sess.run(reduce_lr)
        
        data, target = MNIST.train.next_batch(batch_size)
        
        # Each layer can now be independently updated (could be parallelized)
        if random.random() <= update_prob: # Stochastic updates are possible
            sg_sess.run([layer1_opt], feed_dict={X:data,Y:target})
        if random.random() <= update_prob:
            sg_sess.run([layer2_opt, sg2_opt], feed_dict={X:data,Y:target})
        if random.random() <= update_prob:
            sg_sess.run([layer3_opt, sg3_opt], feed_dict={X:data,Y:target})
        if random.random() <= update_prob:
            _, _, summary = sg_sess.run([layer4_opt, sg4_opt, summary_op], feed_dict={X:data,Y:target})
            sg_train_writer.add_summary(summary, i)
        
        if i % validation_checkpoint == 0:
            Xb, Yb = MNIST.test.next_batch(batch_size)
            summary = sg_sess.run([summary_op], feed_dict={X:Xb,Y:Yb})[0]
            sg_validation_writer.add_summary(summary, i)
    
    # Cleanup summary writers
    sg_train_writer.close()
    sg_validation_writer.close()

# Test using backprop
with backprop_sess.as_default():
    n_batches = int(MNIST.test.num_examples/batch_size)
    test_accuracy = 0
    test_loss = 0
    for _ in range(n_batches):
        Xb, Yb = MNIST.test.next_batch(batch_size)
        batch_accuracy, batch_loss = backprop_sess.run([accuracy, pred_loss], feed_dict={X:Xb,Y:Yb})
        test_loss += batch_loss
        test_accuracy += batch_accuracy
    print (test_loss/n_batches)
    print(test_accuracy/n_batches)

# Test using synthetic gradients
with sg_sess.as_default():
    n_batches = int(MNIST.test.num_examples/batch_size)
    test_accuracy = 0
    test_loss = 0
    for _ in range(n_batches):
        Xb, Yb = MNIST.test.next_batch(batch_size)
        batch_accuracy, batch_loss = sg_sess.run([accuracy, pred_loss], feed_dict={X:Xb,Y:Yb})
        test_loss += batch_loss
        test_accuracy += batch_accuracy
    print (test_loss/n_batches)
    print(test_accuracy/n_batches)

# Cleanup
sg_sess.close()
backprop_sess.close()

