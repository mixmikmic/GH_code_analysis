import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import gym

get_ipython().run_line_magic('matplotlib', 'inline')

import tf_util
import load_policy

# tf_util.initialize()

sess = tf.InteractiveSession()

# with tf.Session():
#     tf_util.initialize()

policy_fn = load_policy.load_policy('./experts/Hopper-v1.pkl')

# None: batch size could vary
x_plh = tf.placeholder(tf.float32, shape=[None, 11])
y_plh = tf.placeholder(tf.float32, shape=[None, 3])

W_var = tf.Variable(tf.zeros([11, 3]))
b_var = tf.Variable(tf.zeros([3]))

sess.run(tf.global_variables_initializer())

y = tf.matmul(x_plh, W_var) + b_var

# cross_entropy = tf.reduce_mean(
#     tf.nn.softmax_cross_entropy_with_logits(labels=y_plh, logits=y))

loss_mse = tf.losses.mean_squared_error(labels=y_plh, predictions=y)

# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss_mse)

train_step

obs, action = None, None
env = gym.make('Hopper-v1')
for _ in range(1000):
    if obs is None and action is None:
        obs = env.reset()
        action = policy_fn(obs[None,:])
    else:
        obs, r, done, _ = env.step(action)
        action = policy_fn(obs[None, :])
    _x = np.array([obs])
    _y = np.array([action.ravel()])
#     batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x_plh: _x, y_plh: _y})

# collect (obs, action) pairs for testing
obs_test, actions_test = [], []
for _ in range(100):
    obs, r, done, _ = env.step(action)
    action = policy_fn(obs[None, :])
    obs_test.append(obs)
    actions_test.append(action.ravel())
obs_test = np.array(obs_test)
actions_test = np.array(actions_test)

obs_test.shape

actions_test.shape

# tf.squared_difference?

# metric_mse = tf.reduce_mean(tf.square(y_plh - y))
metric_mse = tf.reduce_mean(tf.squared_difference(y_plh, y))

# metric_mse = tf.metrics.mean_squared_error(labels=y_plh, predictions=y)

casted = tf.cast(metric_mse, tf.float32)

casted.eval(feed_dict={x_plh: obs_test, y_plh: actions_test})







correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})



