import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) 

print(node1, node2)

sess = tf.Session()
print(sess.run(node1))
print(sess.run([node1, node2]))
print(sess.run(node1+node2))

node3 = tf.add(node1, node2)
print(sess.run(node3))

# Create a Constant op that produces a 1x2 matrix.  
matrix1 = tf.constant([[3., 3.]])

# Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.],[2.]])

# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
product = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
    result = sess.run(product)
    result2 = sess.run(matrix1[0,0] - matrix2[1,0])
    print result
    print result2

matrix3=[[1,3,4],3]
print matrix3[0][2]

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b

with tf.Session() as sess:
    print(sess.run(tf.add(a,b), feed_dict={a:3, b:4.5}))
    print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2,4]}))

W=tf.Variable([.3], tf.float32)
b=tf.Variable([-.3], tf.float32)

x=tf.placeholder(tf.float32)

linear_model=W*x + b

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(linear_model, {x:[1,2,3,4]}))

