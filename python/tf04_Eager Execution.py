
from __future__ import absolute_import, division, print_function
import tensorflow as tf

tf.enable_eager_execution() #必须放在程序开头

tf.executing_eagerly()

x = [[2.]]
m = tf.matmul(x, x)

print("hello,{}".format(m))

a = tf.constant([[1, 2],[3, 4]])
print(a)

b = tf.add(a, 1)
print(b)

# Operator overloading is supported
print(a*b)

# Use NumPy values
import numpy as np

c = np.multiply(a, b)
print(c)

# Obtain numpy value from a tensor:
print(a.numpy())



