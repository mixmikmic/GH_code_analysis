get_ipython().magic('matplotlib notebook')

import os, sys
sys.path.append(os.path.abspath('../../main/python'))

import time
import numpy as np
import thalesians.tsa.pypes as pypes
import thalesians.tsa.visual as visual

pype = pypes.Pype(pypes.Direction.INCOMING, name='liveplotpype', port=22184); pype

liveplot = visual.LivePlot(keep_last_points=10)
liveplot.ax.plot([], label='mid')
liveplot.ax.plot([], [], '^', ms=12, c='green', label='buy')
liveplot.ax.plot([], [], 'v', ms=12, c='red', label='sell')
liveplot.ax.grid()
liveplot.ax.legend(loc='upper right')
liveplot.ax.set_xlabel('time')
liveplot.ax.set_ylabel('price')
liveplot.ax.set_title('prices')
liveplot.refresh()
for i, price in enumerate(pype):
    liveplot.append(i, [price['mid'], price['buy'], price['sell']])

