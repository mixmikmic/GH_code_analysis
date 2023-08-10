import matplotlib.pyplot as plt
import numpy as np
from bokeh import mpl
from bokeh.plotting import output_notebook, show

output_notebook()

x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
y = np.sin(x / 2)
z = np.cos(x / 4)

plt.title("Matplotlib Figure in Bokeh")
plt.plot(x, y, "r-", marker='o')
plt.plot(x, z, "g-x", linestyle="-.")

show(mpl.to_bokeh())

n = 50
x, y, z, s, ew = np.random.rand(5, n)
c, ec = np.random.rand(2, n, 4)
area_scale, width_scale = 500, 5

fig, ax = plt.subplots()
sc = ax.scatter(x, y, c=c,
                s=np.square(s)*area_scale,
                edgecolor=ec,
                linewidth=ew*width_scale)
ax.grid()

show(mpl.to_bokeh())



