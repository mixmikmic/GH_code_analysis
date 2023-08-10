import tensorflow as tf
import numpy as np

x = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,1],
              [0,0,0,0,0,0,0,0,0,0,0,0,1,0],
              [0,0,0,0,0,0,0,0,0,0,0,1,0,0],
              [0,0,0,0,0,0,0,0,0,0,1,0,0,0],
              [0,0,0,0,0,0,0,0,0,1,0,0,0,0],
              [0,0,0,0,0,0,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,0,0,1,0,0,0,0,0,0],
              [0,0,0,0,0,0,1,0,0,0,0,0,0,0],
              [0,0,0,0,0,1,0,0,0,0,0,0,0,0],
              [0,0,0,0,1,0,0,0,0,0,0,0,0,0],
              [0,0,0,1,0,0,0,0,0,0,0,0,0,0],
              [0,0,1,0,0,0,0,0,0,0,0,0,0,0],
              [0,1,0,0,0,0,0,0,0,0,0,0,0,0],
              [1,0,0,0,0,0,0,0,0,0,0,0,0,0]])
x.shape

# where '1's are stored.
x_indices = [[row_i, np.argmax(row)] for row_i, row in enumerate(x)]
x_indices

# what values are stored
x_values = [1]*14
x_values

x_sparse = tf.SparseTensor(
    indices=x_indices,
    values=x_values,
    dense_shape=x.shape)

tf.InteractiveSession()

tf.sparse_tensor_to_dense(x_sparse).eval()

x_sparse.eval()

np.all(x == tf.sparse_tensor_to_dense(x_sparse).eval())

