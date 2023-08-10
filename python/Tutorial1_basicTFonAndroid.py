import tensorflow as tf
print("TensorFlow version: " + tf.__version__)

TUTORIAL_NAME = 'Tutorial1'
MODEL_NAME = 'basicTFonAndroid'
SAVED_MODEL_PATH = '../' + TUTORIAL_NAME+'_Saved_model/'

A = tf.placeholder(tf.float32, shape=[1], name='some_nameA') # input a
B = tf.placeholder(tf.float32, shape=[1], name='some_nameB') # input b

# A tensorflow holder of type variable not used anywhere in this model 
W = tf.Variable(tf.zeros(shape=[1]), dtype=tf.float32, name='some_nameW') # weights

# sum of two matrices element-wise, in our case two numbers in 
AB = tf.add(A, B, name='some_nameAB')

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

sess.run(AB,{A:[2],B:[4]})

tf.train.write_graph(sess.graph_def, SAVED_MODEL_PATH , MODEL_NAME + '_withWeightW_asVariable' + '.pbtxt')

_W = W.eval(sess)
print(_W)

A2 = tf.placeholder(tf.float32, shape=[1], name='modelInputA') # input a
B2 = tf.placeholder(tf.float32, shape=[1], name='modelInputB') # input b

# A tensorflow holder of type constant(not actually used anywhere in this model)
W2 = tf.constant(_W, name='modelWeightW') # weights

# sum of two matrices element-wise, in our case two numbers in 
AB2 = tf.add(A2, B2, name='modelOutputAB')

tf.train.write_graph(sess.graph_def, SAVED_MODEL_PATH , MODEL_NAME + '.pbtxt')
tf.train.write_graph(sess.graph_def, SAVED_MODEL_PATH , MODEL_NAME + '.pb', as_text=False)

import tensorflow as tf
print("TensorFlow version: " + tf.__version__)


TUTORIAL_NAME = 'Tutorial1'
MODEL_NAME = 'basicTFonAndroid'
SAVED_MODEL_PATH = '../' + TUTORIAL_NAME+'_Saved_model/'


A = tf.placeholder(tf.float32, shape=[1], name='some_nameA') # input a
B = tf.placeholder(tf.float32, shape=[1], name='some_nameB') # input b

# A tensorflow holder of type variable not used anywhere in this model 
W = tf.Variable(tf.zeros(shape=[1]), dtype=tf.float32, name='some_nameW') # weights

# sum of two matrices element-wise, in our case two numbers in 
AB = tf.add(A, B, name='some_nameAB')


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


sess.run(AB,{A:[2],B:[4]})


tf.train.write_graph(sess.graph_def, SAVED_MODEL_PATH , MODEL_NAME + '_withWeightW_asVariable' + '.pbtxt')


_W = W.eval(sess)
print(_W)


A2 = tf.placeholder(tf.float32, shape=[1], name='modelInputA') # input a
B2 = tf.placeholder(tf.float32, shape=[1], name='modelInputB') # input b

# A tensorflow holder of type constant(not actually used anywhere in this model)
W2 = tf.constant(_W, name='modelWeightW') # weights

# sum of two matrices element-wise, in our case two numbers in 
AB2 = tf.add(A2, B2, name='modelOutputAB')


tf.train.write_graph(sess.graph_def, SAVED_MODEL_PATH , MODEL_NAME + '.pbtxt')
tf.train.write_graph(sess.graph_def, SAVED_MODEL_PATH , MODEL_NAME + '.pb', as_text=False)

