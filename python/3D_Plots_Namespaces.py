get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Make data
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 - Y**2

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, color='b')

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

fig = plt.figure(figsize=plt.figaspect(0.3))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(X, Y, Z, color='b')
ax = fig.add_subplot(122, projection='3d')
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
U,V = np.meshgrid(u,v)
X = np.cos(U) * np.sin(V)
Y = np.sin(U) * np.sin(V)
Z = np.cos(V)

fig = plt.figure()
ax = plt.axes(projection='3d')
# Plot the surface
ax.plot_surface(X, Y, Z, color='b')

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
X = np.outer(np.cos(u), np.sin(v))
Y = np.outer(np.sin(u), np.sin(v))
Z = np.outer(np.ones(np.size(u)), np.cos(v))

fig = plt.figure()
ax = plt.axes(projection='3d')
# Plot the surface
ax.plot_surface(X, Y, Z, color='b')

# Make data
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, 2 * np.pi, 100)
U,V = np.meshgrid(u,v)
R = 10
r = 4
X = R * np.cos(U) + r*np.cos(U)*np.cos(V)
Y = R * np.sin(U) + r*np.sin(U)*np.cos(V)
Z = r * np.sin(V)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlim([-(R+r), (R+r)])
ax.set_ylim([-(R+r), (R+r)])
ax.set_zlim([-(R+r), (R+r)])
ax.plot_surface(X, Y, Z, color='c')



