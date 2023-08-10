get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

new_saver = tf.train.import_meta_graph('saved_mnist_cnn.ckpt.meta')
new_saver.restore(sess, 'saved_mnist_cnn.ckpt')

tf.get_default_graph().as_graph_def()

x = sess.graph.get_tensor_by_name("input:0")
y_conv = sess.graph.get_tensor_by_name("output:0")
keep_prob = sess.graph.get_tensor_by_name("keep_prob:0")

image_b = mnist.validation.images[159]
plt.imshow(image_b.reshape([28, 28]), cmap='Greys')

image_b = image_b.reshape([1, 784])
result = sess.run(y_conv, feed_dict={x:image_b, keep_prob:1})
print(result)
print(sess.run(tf.argmax(result, 1)))

