import numpy as np

a = np.array([1, 2, 3])
b = np.array([-2, 0, 5])
a.dot(b)

def magnitude1(vector):
    return np.sqrt(np.sum(vector**2))

def magnitude2(vector):
    return np.sqrt(vector.dot(vector))

print('Magnitude1:\t', magnitude1(a))
print('Magnitude2:\t', magnitude2(a))
print('Numpy Norm:\t', np.linalg.norm(a))

