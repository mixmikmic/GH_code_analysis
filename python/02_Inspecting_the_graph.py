# Creating the data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
N = 30
x_vals = (np.linspace(0,10,N)).astype('float32')
y_vals = (2.42 * x_vals + 0.42 + np.random.normal(0,1,N)).astype('float32')

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

loss_summary = tf.scalar_summary("loss_summary", loss) #<--- Definition of ops to be stored
resi_summart = tf.histogram_summary("resi_summart", resi)
merged_summary_op = tf.merge_all_summaries()        #<-----  Combine all ops to be stored
sess = tf.Session()
sess.run(init_op)
writer = tf.train.SummaryWriter("/tmp/dumm/run1", tf.get_default_graph(), 'graph.pbtxt') #<--- Where to store
for e in range(epochs): #Fitting the data for 10 epochs
    sess.run(train_op, feed_dict={x:x_vals, y:y_vals})
    if (e < 5 | e > epochs - 5):
        print("epoch {} {}".format(e, sess.run(loss, feed_dict={x:x_vals, y:y_vals})))
    sum_str = sess.run(merged_summary_op, feed_dict={x:x_vals, y:y_vals}) #<--- Running the graph to produce output
    writer.add_summary(sum_str, e) #<--- writing out the output
res = sess.run([loss, a, b], feed_dict={x:x_vals, y:y_vals})
print(res)
print('Finished all')

loss_t = tf.Graph.get_tensor_by_name(tf.get_default_graph(), 'loss:0')
x_1 = tf.Graph.get_tensor_by_name(tf.get_default_graph(), 'x_data:0')
y_1 = tf.Graph.get_tensor_by_name(tf.get_default_graph(), 'y_data:0')
loss_val = sess.run(loss_t, feed_dict={x_1:x_vals, y_1:y_vals})
print(loss_val)

sess.close()



