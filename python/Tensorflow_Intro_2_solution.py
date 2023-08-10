import tensorflow as tf
import numpy as np

X_data=np.array([[0,0],[1,0],[0,1],[1,1]])
y_data=np.array([[1],[2],[2],[3]])

X=tf.placeholder(shape=(4,2),dtype=tf.float32,name='input')
y=tf.placeholder(shape=(4,1),dtype=tf.float32,name='output')
W=tf.Variable([[1],[1]],dtype=tf.float32,name='weights')
b=tf.Variable([0],dtype=tf.float32,name='bias')
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

with tf.name_scope('Linear_approximation'):
    line_approx=tf.matmul(X,W)+b

with tf.name_scope("loss"):
    loss=tf.reduce_sum(tf.square(line_approx-y,name='loss'))

with tf.name_scope("Optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

with tf.name_scope('Summaries'):
    tf.summary.scalar("Loss",loss)
    tf.summary.histogram("Histogram_loss",loss)
    summary_op=tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./cooler_graphs', sess.graph)
    for i in range(10000):
        _,_,summary=sess.run([loss,optimizer,summary_op],feed_dict={X:X_data,y:y_data})
        writer.add_summary(summary,global_step=i)



