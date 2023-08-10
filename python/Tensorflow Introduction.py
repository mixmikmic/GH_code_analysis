import tensorflow as tf

# Constructing computational graph
a = tf.constant(5)
b = tf.constant(6)
c= a * b

# Run default graph session
with tf.Session() as sess:
    print(c)
    print(sess.run(c))
    print(c.eval())
    print(c)

W1 = tf.ones((2,2))
W2 = tf.Variable(tf.zeros((2,2)), name="weights")
with tf.Session() as sess:
    print(sess.run(W1))
    sess.run(tf.global_variables_initializer())
    print(sess.run(W2))

# initialize from constant values
W = tf.Variable(tf.zeros((2,2)), name="weights")
# initialize from random values
R = tf.Variable(tf.random_normal((2,2)), name="random_weights")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(W))
    print(sess.run(R))


#initialize state variable
state = tf.Variable(0, name="counter")
#initialize new_value as: new_value = state + 1
new_value = tf.add(state, tf.constant(1))
# initialize state as: state = new_value
update = tf.assign(state, new_value)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(state))
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

# initializing constants
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

#intermiate operation
intermed = tf.add(input2, input3)

# final operation
mul = tf.multiply(input1, intermed)

# Calling sess.run(var) on a tf.Session() object retrieves its value. 
# Can retrieve multiple variables simultaneously with sess.run([var1, var2])
with tf.Session() as sess:
    result = sess.run([mul, intermed])
    print(result)

# Define tf.placeholder objects for data entry.
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    # Feed data into computation graph
    print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))



