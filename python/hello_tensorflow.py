import tensorflow as tf

a = tf.constant([[2]])
print(a)

sess = tf.Session()
print(sess.run(a))

sess = tf.Session()
a = tf.constant([2])
print("\nTensor definition: ", a)
print("Value on session:")
print(sess.run(a))
a = tf.constant([[2]])
print("\nTensor definition: ", a)
print("Value on session:")
print(sess.run(a))
a = tf.constant([[2], [1]])
print("\nTensor definition: ", a)
print("Value on session:")
print(sess.run(a))
a = tf.constant([[2, 1]])
print("\nTensor definition: ", a)
print("Value on session:")
print(sess.run(a))
a = tf.constant([[2, 1], [3, 4]])
print("\nTensor definition: ", a)
print("Value on session:")
print(sess.run(a))

a = tf.constant([[2]])
b = tf.constant([[3]])
c = tf.add(a, b)
print("Tensor c: ", c)
sess = tf.Session()
print("Value of c: ", sess.run(c))
sess.close()

scalar = tf.constant([2])
vector = tf.constant([2, 3, 4])
matrix = tf.constant([[2, 3, 4], [5, 6, 7], [8, 9, 10]])
tensor = tf.constant([[[1,2,3], [5, 6, 7], [8, 9, 10]], 
                      [[3, 2, 1], [1, 10, 11], [6, 5 ,7]], 
                      [[9, 7 , 6], [18, 4, 3], [7, 5, 4]]])

with tf.Session() as sess:
    print("\nScalar: \n", sess.run(scalar))
    print("\nVector: \n", sess.run(vector))
    print("\nMatrix: \n", sess.run(matrix))
    print("\nTensor: \n", sess.run(tensor))

matrix_a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matrix_b = tf.constant([[3, 2, 3], [4, 5, 0], [7, 6, 9]])

mat_sum = tf.add(matrix_a, matrix_b)
mat_pro = tf.matmul(matrix_a, matrix_b)
mat_pro_elem = tf.multiply(matrix_a, matrix_b)

with tf.Session() as sess:
    print("\nMatrix addition \n")
    print(sess.run(mat_sum))
    print("\nMatrix multiplication \n")
    print(sess.run(mat_pro))
    print("\nElementwise matrix multiplication\n")
    print(sess.run(mat_pro_elem))

# Graph
state = tf.Variable(0) # define a variable
step = tf.constant(1) # define a unit step
new_value = tf.add(state, step) # define an operation 
update = tf.assign(state, new_value) # update the value of state to new_value using tf.assign()

# Session
with tf.Session() as sess:
    print(sess.run(update))

# Graph
state = tf.Variable(0) # define a variable
step = tf.constant(1) # define a unit step
new_value = tf.add(state, step) # define an operation 
update = tf.assign(state, new_value) # update the value of state to new_value using tf.assign()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(update))

sample = tf.placeholder(tf.int8)
op = tf.multiply(sample, 10)
print("Placeholder : ", sample)
with tf.Session() as sess:
    print(sess.run(op))

sample = tf.placeholder(tf.int8, (3,))
op = tf.multiply(sample, 10)
print("Placeholder : ", sample)
dictionary = {sample: [2, 3, 4]}
with tf.Session() as sess:
    print(sess.run(op, feed_dict=dictionary))

# Graph

# b is a tensor of dimension 10 x 1 with all values as 1
b = tf.Variable(tf.ones((10, 1)))
# W is a tensor with random but uniformly distributed values between -1 and 1 on a dimension of 10 x 10. 
W = tf.Variable(tf.random_uniform((10, 10), -1, 1)) 
# x is a placeholder for a 10 x 1 tensor with type float32. 
x = tf.placeholder(tf.float32, (10, 1))
# calculating f using the above variables
f = tf.add(tf.matmul(W, x), b) # tf.matmul() performs matrix multiplication

# Session

import numpy as np
sess = tf.Session() # define session 
sess.run(tf.global_variables_initializer()) # initialize all the variable to be used
output = sess.run(f, feed_dict={x: np.random.random_sample((10, 1))}) # evaluating f by feeding the 
print(output)
print(output.shape)

