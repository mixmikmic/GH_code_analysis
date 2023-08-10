# Import the tensorflow library.
import tensorflow as tf

# This is not required but useful. See Below!
import numpy as np

# Start every new language with Hello World!
h = tf.constant("Hello World!")
print(h)

with tf.Session() as sess:
    print("1: " + sess.run(h))
    
# NOTE: Session is necessary whenever any evaluation and computations are done.
# Print using the eval function. The eval() takes sessions as an input parameter.
sess1 = tf.Session()
print("2: " + h.eval(session=sess1))
sess.close()

# Creating different types of constants.
# dtype - in case the type of the constant is to be specified.
a = tf.constant(4.0)
b = tf.constant(5, dtype=tf.int32)
with tf.Session() as sess:
    print(sess.run(a), sess.run(b))

x = tf.constant(25.0)
y = tf.constant(5.0)
with tf.Session() as sess:
    print("x = %0.1f"%sess.run(x+y))
    print("y = %0.1f"%sess.run(x+y))
    print("x + y = %0.1f"%sess.run(x+y))
    print("x - y = %0.1f"%sess.run(x-y))
    print("x * y = {:.2f}".format(sess.run(x*y))) # Different way to print.
    print("x / y = %0.1f"%sess.run(x/y))
    print("SQRT(x): %0.1f"%sess.run(tf.sqrt(x)))
    

# What is shape?
m = np.array([[1,2],[3,4]])
n = np.array([[5,6],[7,8]])
print(m.shape, n.shape)

# Broadcast is important concept. In broadcasting, the size of the vector is changed to perform
# a certain operation (IMPORTANT: If possible!)
o = 2
print((m * o), (m * o).shape)

# Try: What happens when o = np.array([[2, 2], [2, 2]]) ?

# Reset.
tf.reset_default_graph()

# A simple example.
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a + b

# Create a session and test.
with tf.Session() as sess2:
    result = sess2.run(c)
    print(result)
    print(result.shape)
    # NOTE: The shape parameter is there because the result is a numpy array.

    # Print all the nodes in the graph.
    print([n.name for n in tf.get_default_graph().as_graph_def().node])

# Try: How many nodes are there when you perform c = (a + b)/2 ?

# Perform basic operations with placeholders.
d = tf.placeholder(tf.float32, shape=None)
e = tf.placeholder(tf.float32, shape=())
sum_1 = a + d
mul_1 = b * e

# Create a session and test.
with tf.Session() as sess2:
    sum_1, mul_1 = sess2.run([sum_1, mul_1], feed_dict={d: 3.0, e:2.0})
    print(sum_1, mul_1)    
    # Try: result = sess2.run(d)
    # What do you see? One of the bad parts about tensorflow.

# Quick example for vector operations.
f = tf.placeholder(tf.float32, shape=(2, 2))
g = tf.placeholder(tf.float32, shape=(3, 3))

sum_2 = a * f
mul_2 = b * g

# print(.shape)
# Create a session and test.
with tf.Session() as sess2:
    # f: Uses python lists.
    # g: Uses numpy arrays.
    sum_2, mul_2 = sess2.run([sum_2, mul_2], feed_dict={f:[[1.0,2.0],[3.0,4.0]],
                                                        g:np.array([[2., 4., 1.],
                                                                    [6., 8., 1.], 
                                                                    [10., 12., 1.]])})
    print(sum_2, mul_2)
    print(sum_2.shape, mul_2.shape)
    # NOTE: Remember broadcast!

