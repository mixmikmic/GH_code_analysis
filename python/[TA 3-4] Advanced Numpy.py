get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt

image = np.array([[[[1],[2],[3]],
                   [[4],[5],[6]],
                   [[7],[8],[9]]]], dtype = np.float32)
print(image.shape)

plt.imshow(image.reshape(3,3), cmap = 'Greys')

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)

img = mnist.train.images[0]
img.shape    # original data with shape (784, )

img = img.reshape(28, 28)
plt.imshow(img, cmap = 'gray')

x = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
v = np.array([1,0,1])
y = np.empty_like(x)

print(x.shape)
print(v.shape)
print(y.shape)

for i in range(4):
    y[i, :] = x[i, :] + v

y

x = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
v = np.array([1,0,1])
vv = np.tile(v, (4,1))

print(vv.shape)

y = x + vv

y

x = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
v = np.array([1,0,1])

y = x + v

y

v = np.array([1,2,3])
w = np.array([4,5])

print(v.shape)
print(w.shape)

np.reshape(v, (3,1))

print(np.reshape(v, (3,1)) * w)

x = np.array([[1,2,3],[4,5,6]])

print(x.shape)

print(x + v)

x.T.shape

print((x.T + w).T)

