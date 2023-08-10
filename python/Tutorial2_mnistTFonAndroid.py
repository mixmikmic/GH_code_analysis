import tensorflow as tf
print('TensorFlow version: ' + tf.__version__)

from tensorflow.examples.tutorials.mnist import input_data as mnist_data
mnist = mnist_data.read_data_sets("../MNIST_data", one_hot=True, validation_size=0)

x_train = mnist.train.images # we will not be using these to feed in data
y_train = mnist.train.labels # instead we will be using next_batch function to train in batches
x_test = mnist.test.images
y_test = mnist.test.labels

print ('We have '+str(x_train.shape[0])+' training examples in dataset')
print ('We have '+str(x_train.shape[1])+' feature points(basically pixels) in each input example')

TUTORIAL_NAME = 'Tutorial2'
MODEL_NAME = 'mnistTFonAndroid'
SAVED_MODEL_PATH = '../' + TUTORIAL_NAME+'_Saved_model/'

LEARNING_RATE = 0.1
TRAIN_STEPS = 1000

X = tf.placeholder(tf.float32, shape=[None, 784], name='modelInput')
Y_ = tf.placeholder(tf.float32, shape=[None, 10])
W = tf.Variable(tf.zeros([784,10]), name='modelWeights')
b = tf.Variable(tf.zeros([10]), name='modelBias')
Y = tf.nn.softmax(tf.matmul(X,W) + b, name='modelOutput')

cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y_ * tf.log(Y), reduction_indices=[1]))
training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)
saver = tf.train.Saver()

for i in range(TRAIN_STEPS+1):
    sess.run(training, feed_dict={X: x_train, Y_: y_train})
    if i%100 == 0:
        print('Training Step:' + str(i) + 
              '  Accuracy =  ' + str(sess.run(accuracy, feed_dict={X: x_test, Y_: y_test})) + 
              '  Loss = ' + str(sess.run(cross_entropy, {X: x_train, Y_: y_train}))
             )
    if i%500 == 0:
        out = saver.save(sess, SAVED_MODEL_PATH + MODEL_NAME + '.ckpt', global_step=i)

tf.train.write_graph(sess.graph_def, SAVED_MODEL_PATH , MODEL_NAME + '.pbtxt')
tf.train.write_graph(sess.graph_def, SAVED_MODEL_PATH , MODEL_NAME + '.pb',as_text=False)

from tensorflow.python.tools import freeze_graph

# Freeze the graph
input_graph = SAVED_MODEL_PATH+MODEL_NAME+'.pb'
input_saver = ""
input_binary = True
input_checkpoint = SAVED_MODEL_PATH+MODEL_NAME+'.ckpt-'+str(TRAIN_STEPS)
output_node_names = 'modelOutput'
restore_op_name = 'save/restore_all'
filename_tensor_name = 'save/Const:0'
output_graph = SAVED_MODEL_PATH+'frozen_'+MODEL_NAME+'.pb'
clear_devices = True
initializer_nodes = ""
variable_names_blacklist = ""

freeze_graph.freeze_graph(
    input_graph,
    input_saver,
    input_binary,
    input_checkpoint,
    output_node_names,
    restore_op_name,
    filename_tensor_name,
    output_graph,
    clear_devices,
    initializer_nodes,
    variable_names_blacklist
)

import tensorflow as tf
print('TensorFlow version: ' + tf.__version__)


from tensorflow.examples.tutorials.mnist import input_data as mnist_data
mnist = mnist_data.read_data_sets("../MNIST_data", one_hot=True, validation_size=0)

x_train = mnist.train.images # we will not be using these to feed in data
y_train = mnist.train.labels # instead we will be using next_batch function to train in batches
x_test = mnist.test.images
y_test = mnist.test.labels

print ('We have '+str(x_train.shape[0])+' training examples in dataset')
print ('We have '+str(x_train.shape[1])+' feature points(basically pixels) in each input example')


TUTORIAL_NAME = 'Tutorial2'
MODEL_NAME = 'mnistTFonAndroid'
SAVED_MODEL_PATH = '../' + TUTORIAL_NAME+'_Saved_model/'


LEARNING_RATE = 0.1
TRAIN_STEPS = 2000


X = tf.placeholder(tf.float32, shape=[None, 784], name='modelInput')
Y_ = tf.placeholder(tf.float32, shape=[None, 10])
W = tf.Variable(tf.zeros([784,10]), name='modelWeights')
b = tf.Variable(tf.zeros([10]), name='modelBias')
Y = tf.nn.softmax(tf.matmul(X,W) + b, name='modelOutput')


cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y_ * tf.log(Y), reduction_indices=[1]))
training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)
saver = tf.train.Saver()


for i in range(TRAIN_STEPS+1):
    sess.run(training, feed_dict={X: x_train, Y_: y_train})
    if i%100 == 0:
        print('Training Step:' + str(i) + 
              '  Accuracy =  ' + str(sess.run(accuracy, feed_dict={X: x_test, Y_: y_test})) + 
              '  Loss = ' + str(sess.run(cross_entropy, {X: x_train, Y_: y_train}))
             )
    if i%500 == 0:
        out = saver.save(sess, SAVED_MODEL_PATH + MODEL_NAME + '.ckpt', global_step=i)
           
        
tf.train.write_graph(sess.graph_def, SAVED_MODEL_PATH , MODEL_NAME + '.pbtxt')
tf.train.write_graph(sess.graph_def, SAVED_MODEL_PATH , MODEL_NAME + '.pb',as_text=False)


from tensorflow.python.tools import freeze_graph

# Freeze the graph
input_graph = SAVED_MODEL_PATH+MODEL_NAME+'.pb'
input_saver = ""
input_binary = True
input_checkpoint = SAVED_MODEL_PATH+MODEL_NAME+'.ckpt-'+str(TRAIN_STEPS)
output_node_names = 'modelOutput'
restore_op_name = 'save/restore_all'
filename_tensor_name = 'save/Const:0'
output_graph = SAVED_MODEL_PATH+'frozen_'+MODEL_NAME+'.pb'
clear_devices = True
initializer_nodes = ""
variable_names_blacklist = ""

freeze_graph.freeze_graph(
    input_graph,
    input_saver,
    input_binary,
    input_checkpoint,
    output_node_names,
    restore_op_name,
    filename_tensor_name,
    output_graph,
    clear_devices,
    initializer_nodes,
    variable_names_blacklist
)

