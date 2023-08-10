import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
N = 30
x_data = (np.linspace(0,10,N)).astype('float32')
y_data = (2.42 * x_data + 0.42 + np.random.normal(0,1,N)).astype('float32')

a = tf.Variable(1.0, name = 'a') #Note that 1.0 is needed
b = tf.Variable(0.01, name = 'b')
x = tf.placeholder('float32', [N], name='x_data')
y = tf.placeholder('float32', [N], name='y_data')


resi = a*x + b - y
loss = tf.reduce_sum(tf.square(resi), name='loss') 
init_op = tf.initialize_all_variables() #Initialization op 
train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)



epochs = 1000
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
    for e in range(5): 
        sess.run(train_op, feed_dict={x:x_data, y:y_data})
    res = sess.run([loss, a, b], feed_dict={x:x_data, y:y_data})
    print(res)
    save_path = saver.save(sess, "checkpoints/model.ckpt")
    print("Model saved in file: %s" % save_path)
get_ipython().system('ls -l checkpoints')

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "checkpoints/model.ckpt")
    res = sess.run([loss, a, b], feed_dict={x:x_data, y:y_data})
    print(res)

definition = tf.get_default_graph().as_graph_def()
type(definition)

tf.train.write_graph(definition, "graphdef/", 'test.pb', as_text=True)
get_ipython().magic('ls -rtl graphdef')
definition.ByteSize()

tf.reset_default_graph() # Clearing the default graph
#saver = tf.train.Saver() # Not working since the graph is empty

from google.protobuf import text_format

# Loading the graph definition
graph_def = tf.GraphDef() 
with open('graphdef/test.pb', "rb") as f:
    text_format.Merge(f.read(), graph_def)
    # Setting the graph definition 
print(graph_def.ByteSize())
with tf.Session() as sess:
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    sess.run(init_op)
    print(tf.all_variables())
    print(sess.graph.get_all_collection_keys())
    saver = tf.train.Saver()
    saver.restore(sess, "checkpoints/model.ckpt")
    res = sess.run([loss, a, b], feed_dict={x:x_data, y:y_data})
    print(res)

from tensorflow.python.client import graph_util
with tf.Session() as sess:
    
    output_graph_def = graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), ['loss', 'y'])
    print('Finished all')

type(['Hallo'])



