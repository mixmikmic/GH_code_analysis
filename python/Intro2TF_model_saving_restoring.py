# for compatibility 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import tensorflow
import tensorflow as tf

# let create some variables
A = tf.Variable(tf.random_normal([ 3, 3]), name="A")
b = tf.Variable(tf.random_normal([ 3, 1]), name="b")
prod = tf.matmul(A,b)

# update value of b
norm = tf.sqrt(tf.reduce_sum(tf.square(prod), 0))
update_op = b.assign(prod/norm)

# init operation
init_op = tf.initialize_all_variables()

saver = tf.train.Saver()

name_of_checkpoint = "model.ckpt"

with tf.Session() as sess:
    sess.run(init_op)
    
    print("Initial values:")
    print(sess.run(A))
    print(sess.run(b))
    
    print("\n Starting updating")
    for _ in range(20):
        sess.run(update_op)
        print(sess.run(b).T)
        
    print("\n Saving model to: " + name_of_checkpoint)
    saver.save(sess,name_of_checkpoint)

saver2restore = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    saver2restore.restore(sess, name_of_checkpoint)
    
    print("Restored values:")
    print(sess.run(A))
    print(sess.run(b))

saver = tf.train.Saver({"eigenv": b})
name_of_checkpoint = 'b.ckpt'

with tf.Session() as sess:
    sess.run(init_op)
    
    print("Initial values:")
    print(sess.run(A))
    print(sess.run(b))
    
    print("\n Starting updating")
    for _ in range(20):
        sess.run(update_op)
        print(sess.run(b).T)
        
    print("\n Saving model to: " + name_of_checkpoint)
    saver.save(sess, name_of_checkpoint)

with tf.Session() as sess:
    sess.run(init_op)
    saver.restore(sess, name_of_checkpoint)
    
    print("Restored values:")
    print(sess.run(A))
    print(sess.run(b))

