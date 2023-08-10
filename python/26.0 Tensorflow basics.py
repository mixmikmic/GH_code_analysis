import tensorflow as tf

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib as mb

hello = tf.constant('Hello World')

type(hello)

x=tf.constant(100)

x

type(x)

sess = tf.InteractiveSession()

sess = tf.Session()

sess.run(hello)

sess.run(x)

type(sess.run(x))

x=tf.constant(2)

y = tf.constant(6)

with tf.Session() as sess:
    print('Operations with Constants')
    print('Addition : ',sess.run(x+y))
    print('Multiplication : ',sess.run(x*y))
    print('Subtration : ',sess.run(x-y))
    print('Division : ',sess.run(x/y))

x = tf.placeholder(tf.int32)

y = tf.placeholder(tf.int32)

x


add = tf.add(x,y)
sub = tf.subtract(x,y)
mul = tf.multiply(x,y)
div = tf.div(x,y)

with tf.Session() as sess :
    print('Operation with plcaeholders')
    print('addition',sess.run(add,feed_dict={x:20,y:30}))
    print ('Subrtraction ',sess.run(sub,feed_dict={x:33,y:44}))
    print ('Multiplication ',sess.run(mul,feed_dict={x:33,y:44}))

a = np.array([[5.0,5.0]])

b = np.array([[2.0],[2.0]])

print ('SHape of array a',a.shape)
print ('SHape of array b',b.shape)

mat1 = tf.constant(a)
mat2 = tf.constant(b)

matrix_multi = tf.matmul(mat2,mat1)

with tf.Session() as sess :
    print('Operation with plcaeholders')
    mult_matirx = sess.run(matrix_multi)
    print(mult_matirx)







