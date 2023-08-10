import tensorflow as tf
my_const = tf.constant([10.0, 4.0], name="my_const")
with tf.Session() as sess:
    print(sess.graph.as_graph_def())
# you will see value of my_const stored in the graph’s definition

import tensorflow as tf
# # created this function to evaluate the expression and print the result.
# like should not be done in production
def run_print(x):
    with tf.Session() as sess:
        print(sess.run(x))


# create a variable with scalar value
a = tf.Variable(2, name="scalar")

# create a variable b as a vector 
b = tf.Variable([2,3],name="vector")

#create a variable as 2X2 matrix
c = tf.Variable([[0,1],[2,3]], name ="matrix")

#create variable W as 784 X 10 tensor, filled with zeros
W = tf.Variable(tf.zeros([784,10]))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    print(sess.run(init))

init_ab = tf.variables_initializer([a, b], name="init_ab")
with tf.Session() as sess:
    print(sess.run(init_ab))

# create variable W as 784 x 10 tensor, filled with zeros
W = tf.Variable(tf.zeros([784,10]))
with tf.Session() as sess:
    print(sess.run(W.initializer))

W = tf.Variable(tf.truncated_normal([700,10]))
with tf.Session() as sess:
    sess.run(W.initializer)
    print(W)

W = tf.Variable(tf.truncated_normal([700,10]))
with tf.Session() as sess:
    sess.run(W.initializer)
    print(W.eval())

W = tf.Variable(10)
W.assign(100)
with tf.Session() as session:
    session.run(W.initializer)
    print(W.eval())

W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as session:
    session.run(assign_op)
    print(W.eval())

# create a variable whose original value is 2
a = tf.Variable(2, name="scalar")

# assign a * 2 to a and call that op a_times_two
a_times_two = a.assign(a * 2)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # Have to inintialize a, becoz a_times_two op depends on the value of a
    print(sess.run(a_times_two))
    print(sess.run(a_times_two))
    print(sess.run(a_times_two))

W = tf.Variable(10)

with tf.Session() as sess:
    sess.run(W.initializer)
    print(sess.run(W.assign_add(2)))
    print(sess.run(W.assign_sub(4)))

W = tf.Variable(10)

session1 = tf.Session()
session2 = tf.Session()

session1.run(W.initializer)
session1.run(W.initializer)

print(session1.run(W.assign_add(10)))
print(session1.run(W.assign_sub(2)))

print(session1.run(W.assign_add(100)))
print(session1.run(W.assign_sub(20)))


session1.close()
session2.close()

W = tf.Variable(tf.truncated_normal([700,10]))
U = tf.Variable(2 * W)

#not safe (but quite common)

W = tf.Variable(tf.truncated_normal([700,10]))
U = tf.Variable(2 * W.initialized_value())

# ensure that W is initialized before its value is used to initialize U -> Safer



