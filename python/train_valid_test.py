import numpy as np

# matrix dimensions
N = 100
M = 20

# train-valid-test ratio, let's use 80-10-10
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 1.0 - train_ratio - valid_ratio  # this is never used

# array indices
train_split = int(train_ratio * N)
valid_split = int(valid_ratio * N)

# create a random matrix
X = np.random.random((N, M))

# create random permutations of row indices
indices = np.random.permutation(range(X.shape[0]))

# split the indices array into train-test-valid
train_indices = indices[:train_split]
valid_indices = indices[train_split:train_split+valid_split]
test_indices = indices[train_split+valid_split:]

# select rows for train-valid-test
X_train = X[train_indices]
X_valid = X[valid_indices]
X_test = X[test_indices]

X_train.shape, X_valid.shape, X_test.shape

nb_samples = 100  # number of samples
nb_features = 20  # number of features
batch_size = 16

X = np.random.random((nb_samples, nb_features))
y = np.random.randint(0, 2, nb_samples)  # random binary labels

batch_indices = np.random.choice(nb_samples, batch_size)

X_batch = X[batch_indices]
y_batch = y[batch_indices]
X_batch.shape, y_batch.shape

import numpy as np
from sklearn.model_selection import train_test_split

nb_samples = 100  # number of samples
nb_features = 20  # number of features
batch_size = 16

X = np.random.random((nb_samples, nb_features))
y = np.random.randint(0, 2, nb_samples)  # random binary labels
X_train, X_test, y_train, y_test = train_test_split(X, y)
y_train.shape, X_test.shape

