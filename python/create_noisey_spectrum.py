get_ipython().magic('matplotlib inline')
import numpy as np
import scipy as sp
import scipy.optimize
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=18, usetex=True)

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rcParams['font.size'] = 20.0

import coronagraph

telescope = coronagraph.Telescope()
print telescope

planet = coronagraph.Planet()
print planet

star = coronagraph.Star()
print star

smart_rad_file = '../planets/F2V_5.e-1fCO2_1.e6H2Volc_1.e10BIF.out_toa.rad'
integration_time = 20.0 # hours
tag = 'F2V'

lam, spec, sig, wlhr, Ahr = coronagraph.smart_observation(smart_rad_file, integration_time, telescope, planet, star, plot=True, tag=tag)

lam, spec, sig = coronagraph.generate_observation(wlhr, Ahr, integration_time, telescope, planet, star, plot=True, tag=tag)

telescope.lammax = 1.6

lam, spec, sig, wlhr, Ahr = coronagraph.smart_observation(smart_rad_file, integration_time, telescope, planet, star, plot=True, tag=tag)

telescope.lammax = 4.5
telescope.temperature = 77.0 # Kelvin (Liquid Nitrogen)

lam, spec, sig, wlhr, Ahr = coronagraph.smart_observation(smart_rad_file, integration_time, telescope, planet, star, plot=True, tag=tag)



