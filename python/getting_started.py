import tensorflow as tf

a = tf.constant(1, name = 'a')
b = tf.constant(2, name = 'b')

c = tf.add(a,b, name = 'c')

print c

with tf.Session() as sess:
    print sess.run(c)

sess =tf.InteractiveSession()
print c.eval()

mat1 = tf.ones((2,2))
mat2 = tf.Variable(tf.zeros((2,2)))





