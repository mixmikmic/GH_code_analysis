import tensorflow as tf
from IPython.display import Image

import numpy as np
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
get_ipython().run_line_magic('matplotlib', 'inline')

def plot_learning_curves(X1, X2, C):
    plt.plot(X1, label='x1')
    plt.plot(X2, label='x2')
    plt.plot(C, label='Cost')
    plt.legend()

Image('./img/tf_optim.PNG')

# tf.train.RMSPropOptimizer(0.02).minimize(objective)
# tf.train.GradientDescentOptimizer(0.002).minimize(objective)
# tf.train.AdamOptimizer(0.3).minimize(objective)
# tf.train.MomentumOptimizer(0.002, 0.9).minimize(objective)
# tf.train.AdadeltaOptimizer(0.1).minimize(objective)
# tf.train.AdagradOptimizer(0.1).minimize(objective)

x1 = tf.Variable(initial_value=4, dtype=tf.float32, name='x1')
x2 = tf.Variable(initial_value=-2, dtype=tf.float32, name='x2')

# cost function
J = 40-(x1**2 + x2**2)

optim = tf.train.RMSPropOptimizer(learning_rate=0.01)
training_op = optim.minimize(-J)

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()

X1, X2, C = [],[],[]
sess.run(init)
n_iter = 2000

for i in range(n_iter):
    _=sess.run(training_op)
    X1.append(sess.run(x1))
    X2.append(sess.run(x2))
    C.append(sess.run(J))
# Get the final values    
print('after {} iterations:'.format(n_iter))
print('Cost: {} at x1={}, x2={}'.format(J.eval(), x1.eval(), x2.eval() ))
plot_learning_curves(X1, X2, C)



