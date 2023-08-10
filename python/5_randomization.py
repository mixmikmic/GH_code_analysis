import tensorflow as tf

a = tf.random_uniform([], -10, 10, seed=2)

with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(a))

with tf.Session() as sess:
    print(sess.run(a))

with tf.Session() as sess:
    print(sess.run(a))

b = tf.random_uniform([], -10, 10, seed=2)
c = tf.random_uniform([], -10, 10, seed=2)

with tf.Session() as sess:
    print(sess.run(b))
    print(sess.run(c))

g = tf.Graph()
with g.as_default():
    tf.set_random_seed(2)
    d = tf.random_uniform([], -10, 10)
    e = tf.random_uniform([], -10, 10)

with tf.Session(graph=g) as sess:
    print(sess.run(d))
    print(sess.run(e))
# If the graph structure changes,this result would be different.

