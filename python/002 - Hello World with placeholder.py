import tensorflow as tf

x = tf.placeholder(tf.string)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Hello World with Placeholder'})

print(output)

y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

with tf.Session() as sess:
    output = sess.run(y, feed_dict={x: 'Test String', y: 123, z: 45.67})

print(output)

with tf.Session() as sess:
    output = sess.run(y, feed_dict={y: 'Hello'})

print(output)



