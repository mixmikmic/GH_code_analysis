import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
get_ipython().magic('matplotlib inline')
plt.rcParams["figure.figsize"] = (13, 8)

def plot_act(i=1.0, actfunc=lambda x: x):
    weights = np.arange(-5, 5, 0.5)
    biases = np.arange(-5, 5, 0.5)
    
    X, Y = np.meshgrid(weights, biases)
    
    os = np.array(
        [actfunc(tf.constant(w * i + b)).eval(session=sess) 
         for w, b in zip(np.ravel(X), np.ravel(Y))])

    Z = os.reshape(X.shape)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1)

# Start a session.
sess = tf.Session()
# Create a simple input of 3 real values.
i = tf.constant([1.0, 2.0, 3.0], shape=[1, 3])
# Create a matrix of weights.
w = tf.random_normal(shape=[3, 3])  # Matrix 3 by 3 of random numbers.
# Create a vector of biases.
b = tf.random_normal(shape=[1, 3])  # Vector of random numbers.
def func(x):
    # Dummy activation function.
    return x
# tf.matmul will multiply the input(i) tensor and the weight(w) tensor then sum the result with the bias(b) tensor.
act = func(tf.matmul(i, w) + b)
# Evaluate the tensor to a numpy array
act.eval(session=sess)

plot_act(1.0, func)

plot_act(1, tf.sigmoid)

act = tf.sigmoid(tf.matmul(i, w) + b)
act.eval(session=sess)

plot_act(1, tf.tanh)

act = tf.tanh(tf.matmul(i, w) + b)
act.eval(session=sess)

plot_act(1, tf.nn.relu)

act = tf.nn.relu(tf.matmul(i, w) + b)
act.eval(session=sess)

