import tensorflow as tf
tf.__version__

hello = tf.constant("Hello, from tensorflow!!")
with tf.Session() as sess:
    print(sess.run(hello))

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

print("node1: ", node1, "node2: ", node2)
print("node3: ", node3)

with tf.Session() as sess:
    print("sess.run(node1, node2): ", sess.run([node1, node2]))
    print("sess.run(node3): ", sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
add = a + b

with tf.Session() as sess:
    print(sess.run(add, feed_dict={a: 3.0, b: 7.0}))
    print(sess.run(add, feed_dict={a: [3.0, 2.5], b: [7.0, 1.5]}))



