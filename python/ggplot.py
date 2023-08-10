from ggplot import aes, diamonds, geom_density, geom_line, geom_step, ggplot, meat
import matplotlib.pyplot as plt

from bokeh import mpl
from bokeh.plotting import output_notebook, show

output_notebook()

g = ggplot(diamonds, aes(x='price', color='cut')) + geom_density()
g.draw()

plt.title("Density ggplot-based plot in Bokeh.")

show(mpl.to_bokeh())

g = ggplot(aes(x='date', y='beef'), data=meat) + geom_line()
g.draw()

plt.title("Line ggplot-based plot in Bokeh.")

show(mpl.to_bokeh())

import numpy as np
import pandas as pd

df = pd.DataFrame({
    "x": range(100),
    "y": np.random.choice([-1, 1], 100)
})
df.y = df.y.cumsum()

g = ggplot(aes(x='x', y='y'), data=df) + geom_step()
g.draw()

plt.title("Step ggplot-based plot in Bokeh.")

show(mpl.to_bokeh())



