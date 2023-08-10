import tensorflow as tf

a = tf.add(2, 3)

print(a)

sess = tf.Session()

print(sess.run(a))  # runs the session

# tf.Session(fetches, feed_dict=None, options=None, run_metadata=None)

# sess.run([a,b]) # computes the value of a and b

sess.close()  # closes the session

with tf.Session() as sess:
    print(sess.run(a))

x = 2
y = 3
op1 = tf.add(x,y)
op2 = tf.multiply(x,y)
op3 = tf.pow(op1,op2)

with tf.Session() as sess:
    print(sess.run(op3))

g = tf.Graph()  # if you want something other than the default graph

with g.as_default():  # making it the default graph
    x = tf.add(2, 3)

sess = tf.Session(graph=g)  # need to pass the graph..
sess.run(x)
sess.close()

