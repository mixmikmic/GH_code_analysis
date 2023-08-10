import tensorflow as tf

# Node-1 Input to Graph
n1 = tf.constant(1)

# Node-2 Input to Graph
n2 = tf.constant(2)

# Node-3 Addition of Previous Two Nodes
n3 = n1 + n2

type(n3)

# Run the Node-3
with tf.Session() as sess:
    result = sess.run(n3)

print(result)

print(n3)

# Current Default Graph
print(tf.get_default_graph())

# Create a New Graph
g = tf.Graph()

print(g)

# Set New Graph as Default Graph
graph_one = tf.get_default_graph()

# New Graph
graph_two = tf.Graph()

# Set Graph 2 as Default Graph
with graph_two.as_default():
    print(graph_two is tf.get_default_graph())

