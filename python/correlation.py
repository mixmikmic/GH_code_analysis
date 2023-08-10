import bokeh
bokeh.sampledata.download()


import time

from numpy import cumprod, linspace, random

from bokeh.sampledata.stocks import AAPL, FB, GOOG, IBM, MSFT
from bokeh.plotting import figure, output_notebook, show

num_points = 300

now = time.time()
dt = 24*3600 # days in seconds
dates = linspace(now, now + num_points*dt, num_points) * 1000 # times in ms
acme = cumprod(random.lognormal(0.0, 0.04, size=num_points))
choam = cumprod(random.lognormal(0.0, 0.04, size=num_points))

output_notebook()

p1 = figure(x_axis_type = "datetime")

p1.line(dates, acme, color='#1F78B4', legend='ACME')
p1.line(dates, choam, color='#FB9A99', legend='CHOAM')

p1.title = "Stock Returns"
p1.grid.grid_line_alpha=0.3

show(p1)

p2 = figure()

p2.scatter(acme, choam, color='#A6CEE3', legend='close')

p2.title = "ACME / CHOAM Correlations"
p2.grid.grid_line_alpha=0.3

show(p2)



