get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")

from matplotlib import pyplot
from random import gauss
from random import seed
from pandas import Series
from pandas.plotting import autocorrelation_plot

seed(1)
series = [gauss(0.0, 1.0) for i in range(1000)]

series = Series(series)

print(series.describe())

# Line plot
series.plot()
pyplot.show()

# histogram plot
series.hist()
pyplot.show()

# autocorrelation
autocorrelation_plot(series)
pyplot.show()



