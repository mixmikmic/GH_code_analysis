import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import os
import matplotlib.pyplot as plt
# display plots in this notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

data="D:/Deep_Learning/TensorFlow/CS 20SI_youtube_video/class_notes/MNIST_data/"
mnist = input_data.read_data_sets(data, one_hot=True)

print("train data :",mnist.train.images.shape)
print("test  data :",mnist.test.images.shape)

def weight_variable(shape,name):
    intial= tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(intial,name=name)

def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=name)

def conv2d(x,W,name):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME',name=name)

def max_pool_2x2(x,name):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name=name)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784],name='x_placeholder')
y_= tf.placeholder(tf.float32, shape=[None, 10],name='y_placeholder')

x_image = tf.reshape(x, [-1, 28,28,1])

# First Layer
W_conv1 = weight_variable([5,5,1,32],'W_conv1')  #  weight shape is [kernel_width, kernel_height, input channel, output channel]
b_conv1 = bias_variable([32],'b_conv1')

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1,'h_conv1_xw') + b_conv1, name='h_conv1')
h_pool1 = max_pool_2x2(h_conv1, 'h_pool1')

# Second Layer
W_conv2 = weight_variable([5,5,32,64], 'W_conv2')
b_conv2 = bias_variable([64],'b_conv2')

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2,'h_conv2_xw') + b_conv2,name='h_conv2')
h_pool2 = max_pool_2x2(h_conv2,'h_pool2')

# Densely Connected Layer
W_fc1 = weight_variable([7*7*64, 1024],'W_fc1')
b_fc1 = bias_variable([1024],'b_fc1')

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64], name='h_pool2_flat')
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name='h_fc1')

keep_prob = tf.placeholder(tf.float32, name='keep_prob')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')

W_fc2 = weight_variable([1024, 10],'W_fc2')
b_fc2 = bias_variable([10],'b_fc2')

y_conv = tf.matmul(h_fc1_drop, W_fc2,name='y_conv') + b_fc2  ## This is logits

# Define Loss
cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv,name='softmax_with_logits'),name='cross_entropy')

# Define global step
model_global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

# Define Optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy,global_step=model_global_step)

correct_prediction =tf.equal(tf.argmax(y_conv,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accurarcy')

# create a saver object
saver = tf.train.Saver()

with tf.name_scope("summaries"):
    tf.summary.scalar('loss',cross_entropy)
    tf.summary.scalar("accuracy",accuracy )
    tf.summary.histogram("histogram loss", cross_entropy)
    summary_op = tf.summary.merge_all()

# launch a session to compute the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs',sess.graph)
    for i in range(20000):
        batch = mnist.train.next_batch(100)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.8})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        loss_batch, _,summary = sess.run([cross_entropy, train_step, summary_op],
                                    feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        writer.add_summary(summary)

    print('test accuracy %g' % accuracy.eval(feed_dict={
         x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    
    # saver  Log file into "./graphs"  fold   and open Tensorboard for seeing it!!
    
    
    writer.close() # close the writer when you`re done using it
    
    # Saves sessions, not graphs
    saver.save(sess, 'checkpoint/MNIST_conv.ckpt',global_step=model_global_step)


# get Default graph Protobuf
tf.get_default_graph().as_graph_def()

# restore the trained model  test part
restore_saver = tf.train.import_meta_graph("checkpoint/MNIST_conv.ckpt-40000.meta")

with tf.Session() as sess:
    restore_saver.restore(sess, "checkpoint/MNIST_conv.ckpt-40000")
    #
    # input test dataset  and  get accuracy
    print(sess.run(tf.get_default_graph().get_tensor_by_name("accurarcy:0"),feed_dict=
                   {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

# using frozen_graph_exporter.py
# Usage  :
#         python3 frozen_grph_exporter.py  /path/to/checkpoint   /name/of/output

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

graph = load_graph("./frozen_model.pb")
for op in graph.get_operations():
        print(op.name)

x_input = graph.get_tensor_by_name("prefix/x_placeholder:0")
keep_prob = graph.get_tensor_by_name("prefix/keep_prob:0")

y_out = graph.get_tensor_by_name("prefix/y_conv:0")

with tf.Session(graph=graph) as sess:
    pred = sess.run(y_out, feed_dict={x_input: mnist.test.images, keep_prob:1.0})
    #print('pred = ',pred)
    pred_np = np.argmax(pred,1)
    target = np.argmax(mnist.test.labels, 1)
    print("target = ",target)
    print("target .shape = ",target.shape[0])
    correct_prediction = np.sum(pred_np == target)
    print("accuracy" , correct_prediction *100/target.shape[0],"%")
    



