get_ipython().magic('matplotlib inline')
import tensorflow as tf

a = tf.constant([5])
b = tf.constant([2])

c = a + b  # Create another TensorFlow object applying the sum (+) operation

with tf.Session() as session:
    result = session.run(c)
    print("The addition of this two constants is: {0}".format(result))

c = a * b

with tf.Session() as session:
    result = session.run(c)
    print("The Multiplication of this two constants is: {0}".format(result))

matrix_a = tf.constant([[2, 3], [3, 4]])
matrix_b = tf.constant([[2, 3], [3, 4]])

first_operation = tf.multiply(matrix_a, matrix_b)
second_operation = tf.matmul(matrix_a, matrix_b)

with tf.Session() as session:
    result = session.run(first_operation)
    print("Element-wise multiplication: \n", result)

    result = session.run(second_operation)
    print("Matrix Multiplication: \n", result)

a = tf.constant(1000)
b = tf.Variable(0)

update = tf.assign(b, a)
with tf.Session() as session:
    session.run(tf.global_variables_initializer()) 
    session.run(update) 
    print("b =", session.run(b))

f = [tf.constant(1), tf.constant(1)]

for i in range(2, 10):
    temp = f[i - 1] + f[i - 2]
    f.append(temp)

with tf.Session() as sess:
    result = sess.run(f)
    print(result)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

c = 2 * a - b

dictionary = {a: [2, 2], b: [3, 4]}
with tf.Session() as session:
    print(session.run(c, feed_dict=dictionary))

a = tf.constant(5.)
b = tf.constant(2.)

c = tf.sin(a)

with tf.Session() as session:
    result = session.run(c)
    print("c = {}".format(result))

