# 1.1 Import tensorflow and other libraries.
import tensorflow as tf
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import pylab

# 1.2 Create input data using NumPy. y = x * 0.1 + 0.3 + noise
x_train = np.random.rand(100).astype(np.float32)
noise = np.random.normal(scale=0.01, size=len(x_train))
y_train = x_train * 0.1 + 0.3 + noise

# Pplot our input data.
pylab.plot(x_train, y_train, '.')

# Create some fake evaluation data
x_eval = np.random.rand(len(x_train)).astype(np.float32)
noise = np.random.normal(scale=0.01, size=len(x_train))
y_eval = x_eval * 0.1 + 0.3 + noise

# 1.3 Buld inference graph.
# Create Variables W and b that compute y_data = W * x_data + b
W = tf.get_variable(shape=[], name='weights')
b = tf.get_variable(shape=[], name='bias')

print(W)
print(b)

# Create a placeholder we'll use later to feed x's into the graph for training and eval.
# shape=[None] means we can put in any number of examples. 
# This is used for minibatch training, and to evaluate a lot of examples at once.
x = tf.placeholder(shape=[None], dtype=tf.float32, name='x')

print(x)

# This is the same as tf.add(tf.mul(W, x), b), but looks nicer
y = W * x + b

# Write the graph so we can look at it in TensorBoard
# Now is a good time to try that
sw = tf.summary.FileWriter('/root/tensorboard/linear', graph=tf.get_default_graph())

# Create a placeholder we'll use later to feed the correct y value into the graph
y_label = tf.placeholder(shape=[None], dtype=tf.float32, name='y_label')
print(y_label)

# 1.4 Build training graph.
loss = tf.reduce_mean(tf.square(y - y_label))  # Create an operation that calculates loss.
optimizer = tf.train.GradientDescentOptimizer(0.5)  # Create an optimizer.
train = optimizer.minimize(loss)  # Create an operation that minimizes loss.

# 'loss', 'optimizer' and 'train' are.
print("loss:", loss)
print("optimizer:", optimizer)
print("train:", train)

# Create an operation to initialize all the variables.
init = tf.global_variables_initializer()
print(init)

# 1.6 Create a session and launch the graph.
sess = tf.InteractiveSession()
sess.run(init)

# Initial W and b values.
print(sess.run([W, b]))

# test that we can compute a y from an x (without having trained anything). 
# x must be a vector, hence [3] not just 3.
x_in = [3]
sess.run(y, feed_dict={x: x_in})

# Calculate accuracy on the evaluation data before training
def eval():
    return sess.run(loss, feed_dict={x: x_eval, y_label: y_eval})
eval()

# Add a Summary Operation so we can visualize the loss in TensorBoard
scalar_summary = tf.summary.scalar('loss', loss)
summary_op = tf.summary.merge([scalar_summary])

# Write the graph so we can look at it in TensorBoard
# Now is a good time to try that
sw = tf.summary.FileWriter('/root/tensorboard/linear', graph=sess.graph)

# 1.7 Perform training.
for step in range(201):
  # Run the training op; feed the training data into the graph
  summary_str, _ = sess.run([summary_op, train], feed_dict={x: x_train, y_label: y_train})
  sw.add_summary(summary_str, step)
  # Watch training happen real time.
  if step % 20 == 0:
    print(step, sess.run([W, b]))

# Write the graph so we can look at it in TensorBoard
# Now is a good time to try that
sw = tf.summary.FileWriter('/root/tensorboard/linear', graph=sess.graph)

# 1.8 Plot the predicted values
pylab.plot(x_train, y_train, '.', label="target")
pylab.plot(x_train, sess.run(y, feed_dict={x: x_train, y_label: y_train}), ".", label="predicted")
pylab.legend()
pylab.ylim(0, 1.0)

# Check accuracy on eval data after training
eval()

def predict(x_in): return sess.run(y, feed_dict={x: [x_in]})

# Save the model
saver = tf.train.Saver()
saver.save(sess, './my_checkpoint.ckpt')

# Current prediction
predict(3)

# Reset the model by running the init op again
sess.run(init) 

# Prediction after variables reinitialized
predict(3)

saver.restore(sess, './my_checkpoint.ckpt')

# Predictions after variables restored
predict(3)



