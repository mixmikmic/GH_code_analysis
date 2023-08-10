get_ipython().magic('matplotlib inline')
from numpy import *
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D

# Make data
x = linspace(-2, 2, 100)
y = linspace(-2, 2, 100)
X, Y = meshgrid(x, y)
Z = X**2 - Y**2

fig = figure()
ax = axes(projection='3d')
ax.plot_surface(X, Y, Z, color='b')

fig = figure()
ax = axes(projection='3d')
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

fig = figure(figsize=figaspect(0.3))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(X, Y, Z, color='b')
ax = fig.add_subplot(122, projection='3d')
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

u = linspace(0, 2*pi, 100)
v = linspace(0, pi, 100)
u,v = meshgrid(u,v)
x = cos(u) * sin(v)
y = sin(u) * sin(v)
z = cos(v)

fig = figure()
ax = axes(projection='3d')
# Plot the surface
ax.plot_surface(x, y, z, color='b')

u = linspace(0, 2*pi, 100)
v = linspace(0, pi, 100)
x = outer(cos(u), sin(v))
y = outer(sin(u), sin(v))
z = outer(ones(size(u)), cos(v))

fig = figure()
ax = axes(projection='3d')
# Plot the surface
ax.plot_surface(x, y, z, color='b')

# Make data
u = linspace(0, 2*pi, 100)
v = linspace(0, 2*pi, 100)
u,v = meshgrid(u,v)
R = 10
r = 4
x = R * cos(u) + r*cos(u)*cos(v)
y = R * sin(u) + r*sin(u)*cos(v)
z = r * sin(v)

fig = figure()
ax = axes(projection='3d')
ax.set_xlim([-(R+r), (R+r)])
ax.set_ylim([-(R+r), (R+r)])
ax.set_zlim([-(R+r), (R+r)])
ax.plot_surface(x, y, z, color='c')



