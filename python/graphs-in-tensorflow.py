import numpy as np
import tensorflow as tf

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)

sum_op = tf.add(x1, x2)
product_op = tf.multiply(x1, x2)

with tf.Session() as session:
    sum_res = session.run(sum_op, feed_dict = {x1 : 2.0, x2 : 1})
    product_res = session.run(product_op, feed_dict = {x1 : 2.0, x2 : 0.5})

sum_res

product_res

with tf.Session() as session:
    sum_res = session.run(sum_op, feed_dict = {x1 : [2.0, 2.0, 1.0], x2 : [0.5, 1, 2]})
    product_res = session.run(product_op, feed_dict = {x1 : [2.0, 4.0], x2 : 0.5})

sum_res

product_res



