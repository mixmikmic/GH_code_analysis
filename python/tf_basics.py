import tensorflow as tf

tf.__version__

# simple hello world in tensorflow

a = tf.constant('Hello, World!')

# start a session

sess = tf.Session()

# run graph in the session
output = sess.run(a)
print(output)

# A is a 0-dimensional tensor
A = tf.constant(123)

# B is a 1-dimensional tensor
B = tf.constant([123,456,789])

# C is a 2-dimensional tensor
C = tf.constant([ [123,456,789], [111,222,333] ])

# Basic constant operations
a = tf.constant(2)
b = tf.constant(3)

# Launch graph in a session to perform operations
with tf.Session() as sess:
    add = sess.run(a+b)
    mul = sess.run(a*b)
    print("Addition with constants:", add)
    print("Multiplication with constants:", mul)

# operations with variable as graph input
a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)

# define the operations
add = tf.add(a, b)
mul = tf.multiply(a, b)

# Launch the graph in the session
with tf.Session() as sess:
    # run the operations
    output = sess.run([add, mul], feed_dict={a: 2, b: 3})
    print(output)

# Just another example
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

with tf.Session() as sess:
    output = sess.run([x, y, z], feed_dict={x: 'I am string.', y: 1234, z: 56.78})
    print(output)
    print(output[0], output[1])

# define a variable
x = tf.Variable(5)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    output = sess.run(x)
    print(output)



