foo = []

bar = foo

foo == bar

foo is bar

id(foo)

id(bar)

# Nesting lists is one way to represent a graph structure like a TensorFlow computation graph.
foo.append(bar) 

# and now foo is inside itself...
foo

# Inport tensorflow, create the graph and define the values
import tensorflow as tf
sess = tf.Session()
graph = tf.get_default_graph()
input_value = tf.constant(1.0)
weight = tf.Variable(0.8)

# We define our operation in the graph
output_value = weight * input_value
op = graph.get_operations()[-1]

op.name

# Generates an operation which will initialize all our variables a finally run it

sess.run(tf.global_variables_initializer())
sess.run(output_value)

# reset previous stuff
tf.reset_default_graph()
sess = tf.Session()
get_ipython().system('rm -rf log_simple_graph')
get_ipython().system('rm -rf log_simple_stat')

x = tf.constant(1.0, name='input')
w = tf.Variable(0.8, name='weight')
y = tf.multiply(w, x, name='output')

# FileWriter needs an output directory.
summary_writer = tf.summary.FileWriter('log_simple_graph', sess.graph)

get_ipython().system('tensorboard --logdir=log_simple_graph')

y_ = tf.constant(0.0)

loss = (y - y_)**2

# Optimizer with a learning rate of 0.025.
optim = tf.train.GradientDescentOptimizer(learning_rate=0.025)

grads_and_vars = optim.compute_gradients(loss)
sess.run(tf.global_variables_initializer())
sess.run(grads_and_vars[0][0])

sess.run(optim.apply_gradients(grads_and_vars))
sess.run(w)

train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)

for i in range(100):
    sess.run(train_step)

sess.run(y)

# reset everything 
tf.reset_default_graph()
get_ipython().system('rm -rf log_simple_stats')

import tensorflow as tf

x = tf.constant(1.0, name='input')
w = tf.Variable(0.8, name='weight')
y = tf.multiply(w, x, name='output')
y_ = tf.constant(0.0, name='correct_value')
loss = tf.pow(y - y_, 2, name='loss')
train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)

for value in [x, w, y, y_, loss]:
    tf.summary.scalar(value.op.name, value)

summaries = tf.summary.merge_all()

session = tf.Session()
summary_writer = tf.summary.FileWriter('log_simple_stats', session.graph)

session.run(tf.global_variables_initializer())
for i in range(100):
    summary_writer.add_summary(session.run(summaries), i)
    session.run(train_step)

summary_writer.close()

get_ipython().system('tensorboard --logdir=log_simple_stats')



