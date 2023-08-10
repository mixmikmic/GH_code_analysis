import numpy as np

def f(x):
    if x<=1/2:
        return x*2
    if x>1/2:
        return x*2-1

x = 1/10
for i in range(80):
    print(x)
    x = f(x)

X = np.array([0.85, 0.1, 0.05, 0.]).reshape(1, 4)

Y = np.array([[0.9, 0.07, 0.02, 0.01],
              [0, 0.93, 0.05, 0.02],
              [0, 0, 0.85, 0.15],
              [0, 0, 0, 1.]])

np.dot(X, Y)

X @ Y

X * Y

a = np.array([2,2])

# vector 1-norm
np.linalg.norm(a, ord=1)

# vector 1-norm
np.linalg.norm(a, ord=2)

m = np.array([[5,3],[0,4]])

# matrix 1-norm
np.linalg.norm(m, ord=1)

# vector 2-norm
np.linalg.norm(m, ord=2)

a = np.array([1,2,3,4])
b = np.array([1,5])

a - 3

a.reshape((4,1)) - 3

a.reshape((2,2)) - 3

#a - b #cannot broadcast
a.reshape((4,1)) - b.reshape((1, 2)) # both a and b are broadcasted to 4x2

a.reshape((2, 2)) - b.reshape((1, 2))



