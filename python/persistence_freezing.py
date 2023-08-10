import tensorflow as tf
import numpy as np
N = 30
x_data = (np.linspace(0,10,N)).astype('float32')
y_data = (2.42 * x_data + 0.42 + np.random.normal(0,1,N)).astype('float32')

a = tf.Variable(1.0, name = 'a') #Note that 1.0 is needed
b = tf.Variable(0.01, name = 'b')
x = tf.placeholder('float32', [N], name='x_data')
y = tf.placeholder('float32', [N], name='y_data')


resi = a*x + b - y
loss = tf.reduce_sum(tf.square(resi), name='loss') # <-- We have to give it a name to access it later
init_op = tf.initialize_all_variables() #Initialization op 
train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

with tf.Session() as sess:
    sess.run(init_op)
    for e in range(5): 
        sess.run(train_op, feed_dict={x:x_data, y:y_data})
    res = sess.run([loss, a, b], feed_dict={x:x_data, y:y_data})
    from tensorflow.python.client import graph_util
    output_graph_def = graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), ['loss', 'y_data'])
    with tf.gfile.GFile('graphdef/linear_regression.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))

    

import tensorflow as tf
import numpy as np
tf.reset_default_graph() #Fresh graph, simulates a restart of the kernel 
with tf.gfile.FastGFile('/home/dueo/workspace/dl_tutorial/tensorflow/LinearRegression/graphdef/linear_regression.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
    print(graph_def.ByteSize())
    #print(graph_def)
    with tf.Session() as sess:
        loss = sess.graph.get_tensor_by_name('loss:0')
        x_data = np.zeros((30), dtype='float32')
        y_data = np.ones((30), dtype='float32')
        res = sess.run(loss, feed_dict={'x_data:0':x_data, 'y_data:0':y_data})
        print("Loss {}".format(res))
    



