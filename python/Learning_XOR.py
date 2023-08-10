import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])
W = np.array([[1,1],[1,1]])
c = np.array([[0],[-1]])
w = np.array([[1],[-2]])
b = 0

x_coords = [0,0,1,1]
y_coords = [0,1,0,1]
plt.scatter(x_coords, y_coords)
plt.show()

XW = X.dot(W)
XW_c = np.add(c.T, XW)
h = np.maximum(0, XW_c)
y_hat = h.dot(w) + b
y_hat

y_hat = np.maximum(0, np.add(c.T, X.dot(W))).dot(w) + b
y_hat

y == y_hat

x_coords = [0,1,1,2]
y_coords = [0,0,0,1]
plt.scatter(x_coords,y_coords)
plt.plot(x_coords,y_coords)
plt.show()

