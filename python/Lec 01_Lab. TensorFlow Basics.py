# import tensorflow
import tensorflow as tf

#create a Constant op(operation)
hello = tf.constant('Hello, TensorFlow!')

#Start tf session
sess = tf.Session()

#run operation
print(sess.run(hello))

# Tensor information ==> run을 하기 전에는 아무 역할도 하지 않음
print(hello)

sess = tf.Session()

a = tf.constant(2)
b = tf.constant(3)

c = a + b

#print out operation
print(c)

#print out the result of operation
print(sess.run(c))

#Basic operations

sess = tf.Session()

a = tf.constant(2)
b = tf.constant(3)

with tf.Session() as sess:
    print("a=2, b=3")
    print("Addition with constants: %i" % sess.run(a + b))
    print("Multiplication with constants: %i" % sess.run(a * b))    

#Placeholder

sess = tf.Session()

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)
mul = tf.mul(a, b)

with tf.Session() as sess:
    print("a=2, b=3")
    print("Addition with constants: %i" % sess.run(add, feed_dict={a:2, b:3}))
    print("Multiplication with constants: %i" % sess.run(mul, feed_dict={a:2, b:3})) 



