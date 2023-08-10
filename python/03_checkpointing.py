import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
N = 30
x_val = (np.linspace(0,10,N)).astype('float32')
y_val = (2.42 * x_val + 0.42 + np.random.normal(0,1,N)).astype('float32')

tf.reset_default_graph()
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
        sess.run(train_op, feed_dict={x:x_val, y:y_val})
    res = sess.run([loss, a, b], feed_dict={x:x_val, y:y_val})
    print(res)
    save_path = saver.save(sess, "checkpoints/model.ckpt") #Weights and meta file
    print("Model saved in file: %s" % save_path)
get_ipython().system('ls -l checkpoints')

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "checkpoints/model.ckpt") #Loading the weights
    res = sess.run([loss, a, b], feed_dict={x:x_val, y:y_val})
    print(res)

tf.reset_default_graph() #Start from scratch
saver = tf.train.import_meta_graph('checkpoints/model.ckpt.meta') #Reconstruct the graph
graph = tf.get_default_graph()

# To get the correct names, one can use
#ops = tf.get_default_graph().get_operations()
#for i in ops:
#    print(i.name)
x = graph.get_tensor_by_name('x_data:0')
y = graph.get_tensor_by_name('y_data:0')
loss = graph.get_tensor_by_name('loss:0')
with tf.Session() as sess:
    saver.restore(sess, "checkpoints/model.ckpt")
    res = sess.run([loss], feed_dict={x:x_val, y:y_val})
    print(res)



