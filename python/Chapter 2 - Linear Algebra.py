import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

A = np.array([[1, 2], [4, 3]])

w, v = np.linalg.eig(A)

def normalize(u):
    for i, x in enumerate(u):
        norm = np.sqrt(sum([val**2 for val in x]))
        for j, y in enumerate(x):
            u[i][j] = y / norm

normalize(v)

x_values = np.linspace(-1, 1, 1000)
y_values = np.array([np.sqrt(1 - (x**2)) for x in x_values])

x_values = np.concatenate([x_values, x_values])
y_values = np.concatenate([y_values, -y_values])

u = np.array([[x, y] for x, y in zip(x_values, y_values)])

trans = np.dot(A, u.T).T

fig = plt.figure()
arrow_dir = np.array([[0, 0, v[0][0], v[0][1]], [0, 0, v[1][0], v[1][1]]])
X, Y, U, V = zip(*arrow_dir)
ax = fig.add_subplot(1, 2, 1)
plt.plot(x_values, y_values, '.')
ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
ax.axis('equal')

plt.xlim([-2, 2])
plt.ylim([-2, 2])


ax = fig.add_subplot(1, 2, 2)
ax.axis('equal')
plt.plot(trans[:, 0], trans[:, 1], '.')

plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.show()

