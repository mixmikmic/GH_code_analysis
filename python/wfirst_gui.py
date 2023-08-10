import sys
import os
import numpy as np
import matplotlib
from matplotlib import style
style.use('ggplot')  # see http://matplotlib.org/users/style_sheets.html for info on matplotlib styles
matplotlib.use('nbagg')  # required for interactive plotting
import matplotlib.pyplot as plt
sys.path.append('/git/pandeia/engine')  # required to access installed pandeia codebase

# the first pandeia import is required to run the GUI. the others are provided to allow manual
# running of calculations and loading/saving of inputs or results.
from pandeia.engine.nb_utils import WFIRST_gui
from pandeia.engine.perform_calculation import perform_calculation
from pandeia.engine.io_utils import read_json, write_json

g = WFIRST_gui()
g.display

c2 = g.calc_results
i2 = g.calc_input

from pandeia.engine.calc_utils import build_default_source
s = build_default_source()

s['spectrum']['normalization']['norm_fluxunit'] = 'abmag'
s['spectrum']['normalization']['norm_flux'] = 24.
s['shape']['geometry'] = 'sersic'
s['shape']['sersic_index'] = 1.  # exponential disk
s['shape']['major'] = 0.4  # major axis in arcseconds
s['shape']['minor'] = 0.1  # minor axis in arcseconds
s['position']['y_offset'] = 1.0  # offset in arcseconds
s['position']['orientation'] = 23.  # Orientation relative to horizontal in degrees

i2['scene'].append(s)

r = perform_calculation(i2)

g.calc_results = r



