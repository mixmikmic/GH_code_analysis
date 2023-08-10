get_ipython().magic('matplotlib inline')
import numpy
import matplotlib.pyplot as plt

x = numpy.linspace(-5, 5, 100)
# print x
y = x**2 + 2 * x + 3
# print y
plt.plot(x, y, 'r')

plt.plot(x, y, 'r--')
plt.xlabel("time (s)")
plt.ylabel("Tribble Population")
plt.title("Tribble Growth")
plt.xlim([0, 4])
plt.ylim([0, 25])

x = numpy.linspace(-1, 1, 100)
y = numpy.linspace(-1, 1, 100)
X, Y = numpy.meshgrid(x, y)
F = numpy.sin(X**2) + numpy.cos(Y**2)
plt.pcolor(X, Y, F)
plt.show()

color_map = plt.get_cmap("Oranges")
plt.gcf().set_figwidth(plt.gcf().get_figwidth() * 2)
plt.subplot(1, 2, 1, aspect="equal")
plt.pcolor(X, Y, F, cmap=color_map)

plt.colorbar()
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Tribble Density ($N/m^2$)")

plt.subplot(1, 2, 2, aspect="equal")
plt.pcolor(X, Y, X*Y, cmap=plt.get_cmap("RdBu"))
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Klingon Population ($N$)")
plt.colorbar()

plt.autoscale(enable=True, tight=False)
plt.show()

x = numpy.linspace(-5, 5, 100)

fig = plt.figure()
axes = fig.add_subplot(1, 1, 1)
growth_curve = axes.plot(x, x**2 + 2 * x + 3, 'r--')
axes.set_xlabel("time (s)")
axes.set_ylabel("Tribble Population")
axes.set_title("Tribble Growth")
axes.set_xlim([0, 4])
axes.set_ylim([0, 25])

x = numpy.linspace(-1, 1, 100)
y = numpy.linspace(-1, 1, 100)
X, Y = numpy.meshgrid(x, y)

fig = plt.figure()
fig.set_figwidth(fig.get_figwidth() * 2)

axes = fig.add_subplot(1, 2, 1, aspect='equal')
tribble_density = axes.pcolor(X, Y, numpy.sin(X**2) + numpy.cos(Y**2), cmap=plt.get_cmap("Oranges"))
axes.set_xlabel("x (km)")
axes.set_ylabel("y (km)")
axes.set_title("Tribble Density ($N/km^2$)")
cbar = fig.colorbar(tribble_density, ax=axes)
cbar.set_label("$1/km^2$")

axes = fig.add_subplot(1, 2, 2, aspect='equal')
klingon_population_density = axes.pcolor(X, Y, X * Y + 1, cmap=plt.get_cmap("RdBu"))
axes.set_xlabel("x (km)")
axes.set_ylabel("y (km)")
axes.set_title("Klingon Population ($N$)")
cbar = fig.colorbar(klingon_population_density, ax=axes)
cbar.set_label("Number (thousands)")

plt.show()



