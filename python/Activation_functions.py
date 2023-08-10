from math import e
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

def sigmoid(z):
    y = 1 / (1 + e ** -z)
    return y

demo_list = np.arange(-10, 11, 1)  # creates array [-10, 9, .., 9, 10]
out = [sigmoid(x) for x in demo_list]

plt.xticks(np.arange(21), demo_list)
plt.plot(out)
plt.plot([0,20], [.5,.5], color='black', label="x")
plt.title("Sigmoid Function")
plt.show()

plt.title("RELU Function")
plt.xticks(np.arange(21), demo_list)
plt.plot(np.maximum(demo_list, 0))
plt.show()



