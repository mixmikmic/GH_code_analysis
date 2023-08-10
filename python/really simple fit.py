import datetime
datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt

x = [1, 1.5, 2, 4, 8, 17]
y1 = [0.5, 0.75, 1, 2, 4, 8]
y2 = [1, 1.5, 1.75, 3.25, 6, 16]

from sherpa import ui

ui.load_arrays(1, x, y1)
ui.load_arrays(2, x, y2)

ui.set_stat('leastsq')
ui.get_data_plot_prefs()['yerrorbars'] = False

ui.plot_data(1)

ui.plot_data(2)

ui.plot_data(1)
ui.plot_data(2, overplot=True)

ui.set_source(1, ui.polynom1d.m1)

print(m1)

ui.plot_fit(1)

m1.c1 = 1
ui.plot_fit(1)

ui.fit(1)

ui.plot_fit(1)

ui.thaw(m1.c1)
ui.fit(1)
ui.plot_fit(1)

ui.plot_resid(1)

np.abs(ui.get_resid_plot(1).y).max()

ui.plot_resid(1)
_ = plt.ylim(-0.17, 0.17)

ui.set_source(2, ui.polynom1d.m2)
# I could have said ui.thaw(m2.c1), but the parameter attributes can also be changed directly
m2.c1.frozen = False
ui.fit(2)
ui.plot_fit_resid(2)

resid2 = np.asarray(y2) - m2(x)

resid2.min()

resid2.max()

ui.load_arrays(3, x, y2)
ui.set_source(3, ui.polynom1d.m3)
ui.thaw(m3.c0, m3.c1, m3.c2)  # thaw can be applied to multiple parameters
ui.fit(3)
ui.plot_fit_resid(3)

ui.plot_data(2)
ui.plot_source(2, overplot=True)
ui.plot_source(3, overplot=True)

def ingredients(nservings):
    """How much oatmeal and water do I need?
    
    Parameters
    ----------
    nservings: int
        The number of servings
        
    Returns
    -------
    (oatmeal, water_lin, water_quad)
        The number of cups of oatmeal and water. The water values are
        calculated using a linear and quadratic approximation respectively.
    """
    
    x = [nservings * 1.0]
    oats = m1(x)[0]
    water_l = m2(x)[0]
    water_q = m3(x)[0]
    return (oats, water_l, water_q)

ingredients(1)

ingredients(3)

ingredients(4)

