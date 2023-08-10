"""practical deep learning, chapter 03
1-D graph plot for activation functions and their derivatives"""

import tensorflow 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def sigmoid_tf(x):
    return tf.div(tf.constant(1.0),
                  tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))

def sigmoid_tf_prime(x):
    return tf.multiply(sigma(x), tf.subtract(tf.constant(1.0), sigma(x)))

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))

def leaky_relu(x, alpha):
    return np.maximum(np.zeros(len(x)), x) +  alpha * np.minimum(np.zeros(len(x)), x)

def leaky_relu_prime(x, alpha):
    return np.hstack([alpha * np.ones(len(np.where(x<0)[0])), np.ones(len(np.where(x>=0)[0]))])

def tanh(x):
    return np.tanh(x)
    
def tanh_prime(x):
    return (1 - (x ** 2))

def relu(x):
    return np.maximum(np.zeros(len(x)), x)

def relu_prime(x):
    return np.hstack([np.zeros(len(np.where(x<0)[0])), np.ones(len(np.where(x>=0)[0]))])

def step_function(x):
    y = (x/abs(x) + 1) * 0.5
    return y

def identity(x):
    return x

def identity_prime(x):
    return np.ones(len(x))

x = np.arange(-3,3,0.1)
a = 0.4

# tanh
plt.plot(x, tanh(x))
plt.grid(True)

# tanh_prime
plt.plot(x, tanh_prime(x))
plt.grid(True)

# sigmoid
plt.plot(x, sigmoid(x))
plt.grid(True)

# sigmoid_prime
plt.plot(x, sigmoid_prime(x))
plt.grid(True)

# Identity function
plt.plot(x, identity(x))
plt.grid(True)

# Identity_prime function
plt.plot(x, identity_prime(x))
plt.grid(True)

# ReLu
plt.plot(x, relu(x))
#plt.plot(x, np.maximum(np.zeros(len(x)), x))
plt.grid(True)

# relu_prime
plt.plot(x, relu_prime(x))
plt.grid(True)

# leaky_relu
plt.plot(x, leaky_relu(x,a))
plt.grid(True)

# leaky_relu
plt.plot(x, leaky_relu_prime(x,a))
plt.grid(True)

# step function
plt.plot(x, step_function(x))
plt.grid(True)

