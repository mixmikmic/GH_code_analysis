import tensorflow as tf

print("Tensorflow version: {}".format(tf.__version__))

# Simple hello world using TensorFlow
# 
# Create a Constant op (op = operation)
# The op is added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.

hello = tf.constant('Hello, TensorFlow!')
print(hello)


# Start tf session
sess = tf.InteractiveSession()

# Run graph
print(sess.run(hello).decode())



