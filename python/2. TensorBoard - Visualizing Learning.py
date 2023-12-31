import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a,b)
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs',sess.graph)
    print(sess.run(x))
    
# close the writer when you are done using it
writer.close()

a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
x = tf.add(a,b, name='add')
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs',sess.graph)
    print(sess.run(x))
    
# close the writer when you are done using it
writer.close()

