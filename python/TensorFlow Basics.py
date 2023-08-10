import tensorflow as tf
import seaborn as sns

hello = tf.constant('Hello World')

type(hello)

x = tf.constant(100)

type(x)

sess = tf.Session()

sess.run(hello)

type(sess.run(hello))

sess.run(x)

type(sess.run(x))

x = tf.constant(2)
y = tf.constant(3)

with tf.Session() as sess:
    print('Operations with Constants')
    print('Addition',sess.run(x+y))
    print('Subtraction',sess.run(x-y))
    print('Multiplication',sess.run(x*y))
    print('Division',sess.run(x/y))

x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)

x

type(x)

add = tf.add(x,y)
sub = tf.subtract(x,y)
mul = tf.multiply(x,y)

d = {x:20,y:30}

with tf.Session() as sess:
    print('Operations with Constants')
    print('Addition',sess.run(add,feed_dict=d))
    print('Subtraction',sess.run(sub,feed_dict=d))
    print('Multiplication',sess.run(mul,feed_dict=d))

import numpy as np
# Make sure to use floats here, int64 will cause an error.
a = np.array([[5.0,5.0]])
b = np.array([[2.0],[2.0]])

a

a.shape

b

b.shape

mat1 = tf.constant(a)

mat2 = tf.constant(b)

matrix_multi = tf.matmul(mat1,mat2)

with tf.Session() as sess:
    result = sess.run(matrix_multi)
    print(result)

