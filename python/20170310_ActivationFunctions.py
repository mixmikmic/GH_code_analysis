import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
x = np.arange(-5, 5, 0.01)
y = 1 / (1 + np.exp(-x))
plt.plot(x,y)
plt.title('Logistic Activation Function')
plt.xlabel('Input')
plt.ylabel('Output');

x = np.arange(-5, 5, 0.01)
y = (2 / (1 + np.exp(-2*x)))-1
plt.plot(x,y)
plt.title('Tanh Activation Function')
plt.xlabel('Input')
plt.ylabel('Output');

x = np.arange(-5, 5, 0.01)
y = np.arctan(x)
plt.plot(x,y)
plt.title('Arctan Activation Function')
plt.xlabel('Input')
plt.ylabel('Output');

x = np.arange(-5, 5, 0.01)
z = np.zeros(len(x))
y = np.maximum(z,x)
plt.plot(x,y)
plt.title('ReLU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output');

x = np.arange(-5, 5, 0.01)
y = np.log(1+np.exp(x))
plt.plot(x,y)
plt.title('Softplus Activation Function')
plt.xlabel('Input')
plt.ylabel('Output');

