get_ipython().run_line_magic('matplotlib', 'inline')

import pandas
from altair import *

from matplotlib import pyplot as plt

from astroML.datasets import fetch_great_wall

#------------------------------------------------------------
# Fetch the great wall data
gw = fetch_great_wall()
#------------------------------------------------------------
data = pandas.DataFrame()
data['y (Mpc)'] = gw[:, 1]
data['x (Mpc)'] = gw[:, 0]


chart = Chart(data, max_rows = 100000).mark_circle(size=5, color = 'black', opacity=0.5).encode(
    y=Y('x (Mpc)', scale = Scale(domain=(-365, -175 ))),
    x=X('y (Mpc)')
).configure_cell(
    width=800,
    height=350
).configure_axis(
    grid=False,
    axisWidth=1,
    tickWidth=1,
    labels=True,        
)

chart.display()

