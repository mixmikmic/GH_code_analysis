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

# set gpu device
os.environ['CUDA_VISIBLE_DEVICES']='0' 

#   prepare the data
data="D:/Deep_Learning/TensorFlow/CS 20SI_youtube_video/class_notes/MNIST_data/"
mnist = input_data.read_data_sets(data, one_hot=True)
print("train data :",mnist.train.images.shape)
print("test  data :",mnist.test.images.shape)

x_ = tf.placeholder(tf.float32,[None, 784], name='Data')
y_ = tf.placeholder(tf.float32, [None, 10], name='s_label')
y_soft_target = tf.placeholder(tf.float32, [None, 10], name='sf_target')
T = tf.placeholder(tf.float32, name='tempalate')

W1 = tf.Variable(tf.truncated_normal([784, 800]),name='s_W1')
b1 = tf.Variable(tf.zeros([800]),name='s_b1')
h_1 = tf.matmul(x_, W1,name='h_1')
h_1 = tf.nn.relu(tf.add(h_1, b1,name='h_1_b'))

W2 = tf.Variable(tf.truncated_normal([800,300]),name='s_W2')
b2 = tf.Variable(tf.zeros([300]),name='s_b2')
h_2 = tf.matmul(h_1, W2, name='h_2')
h_2 = tf.nn.relu(tf.add(h_2, b2 , name='h_2_b'))

W3 = tf.Variable(tf.truncated_normal([300,10]), name='s_W3')
b3 = tf.Variable(tf.zeros([10]), name='s_b3')
logits = tf.matmul(h_2, W3, name='s_logits')
logits = tf.add(logits, b3, name='s_logits_b')        

# define params
alpha = 0.1
logits_pre = logits - tf.reduce_max(logits)
# define hard Loss
#hard_loss = tf.nn.softmax(logits,name='hl_softmax')
#hard_loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(hard_loss),reduction_indices=[1],name='hd_loss')) 
hard_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_pre,labels=y_, name='hd_loss'))

# define soft Loss
#soft_loss = tf.nn.softmax(logits/T, name='sf_softmax')
#soft_loss = tf.reduce_mean(-tf.reduce_sum(y_soft_target * tf.log(soft_loss), reduction_indices=[1], name='sf_loss'))
soft_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_pre, labels=y_soft_target, name='soft_loss'))

# regularization
reg_loss = tf.reduce_sum(tf.abs(W1)) + tf.reduce_sum(tf.abs(W2)) #+  tf.reduce_sum(tf.abs(W3))

# define Loss
Loss = hard_loss * alpha + soft_loss *(1-alpha) * tf.pow(T,2)# + 0.0001*reg_loss

# Define global step
model_global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step_s')

# Define Optimizer
train_step = tf.train.AdamOptimizer(1e-2).minimize(Loss,global_step=model_global_step)

correct_prediction =tf.equal(tf.argmax(logits,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accurarcy_s')

# saver model
saver = tf.train.Saver()

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
#for op in graph.get_operations():
       # pass
        #print(op.name)

# teacher input tensor
x_input = graph.get_tensor_by_name("prefix/x_placeholder:0")
keep_prob = graph.get_tensor_by_name("prefix/keep_prob:0")
# teacher predict tensor
y_out = graph.get_tensor_by_name("prefix/y_conv:0")

# set T params
params_t = 1

sess_teacher = tf.Session(graph=graph)
pred = sess_teacher.run(y_out, feed_dict={x_input: mnist.test.images, keep_prob:1.0})
pred_np = np.argmax(pred,1)
target = np.argmax(mnist.test.labels, 1)
correct_prediction = np.sum(pred_np == target)
print("teacher network accuracy =" , correct_prediction /target.shape[0])

tf.device("/gpu:0")
with tf.Session() as sess_student:
    sess_student.run(tf.global_variables_initializer())
    for i in range(11000):
        batch = mnist.train.next_batch(100)
        
        # teacher soft_target    
        soft_target = sess_teacher.run(y_out, feed_dict={x_input: batch[0], keep_prob:1.0})
        soft_target = tf.nn.softmax(soft_target/params_t)
        
        # student train processing
        train_step.run(feed_dict={x_ :batch[0], y_: batch[1], T : params_t, y_soft_target:soft_target.eval() })
        
        if i % 200 == 0:
            hd_loss, sf_loss, loss_num, train_accuracy = sess_student.run([hard_loss, soft_loss ,Loss, accuracy], 
                                                        feed_dict={x_:batch[0],  y_:batch[1],
                                                                   T:1.0, y_soft_target:soft_target.eval()  }) 
            print('step %d, training accuracy %g , loss = %g , hard_loss = %g, soft_loss = %g' % 
                       (i, train_accuracy, loss_num, hd_loss, sf_loss ))
        if i % 1000 == 0:
            print('test accuracy %g' % sess_student.run(accuracy,feed_dict={
                                    x_: mnist.test.images, y_: mnist.test.labels, T: 1.0}))
            
    print('Finally - test accuracy %g' % sess_student.run(accuracy,feed_dict={
                                    x_: mnist.test.images, y_: mnist.test.labels, T: 1.0}))
           
    
sess_teacher.close()




