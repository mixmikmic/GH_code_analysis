import matplotlib.pyplot as plt
import numpy as np
import plotly

# Turn off matplotlib interactive mode
plt.ioff()

# Turn on offline mode (so we do everything locally, i.e. without a Plotly account)
plotly.offline.init_notebook_mode() # run at the start of every notebook

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

pfig = plotly.tools.mpl_to_plotly(fig)

plotly.offline.iplot(pfig)

x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
y = np.sin(x / 2)
z = np.cos(x / 4)

plt.title("Matplotlib Figure in Bokeh")
plt.plot(x, y, "r-", marker='o')
plt.plot(x, z, "g-x", linestyle="-.")

pfig = plotly.tools.mpl_to_plotly(plt.gcf())
plotly.offline.iplot(pfig)

