get_ipython().magic('matplotlib inline')
import numpy as np
import glob
import hera_qm
from hera_qm.data import DATA_PATH
import os
import fnmatch

# get files
files = sorted(glob.glob(os.path.join(DATA_PATH, "*.first.calfits")))

f = fnmatch.filter(files, '*/zen.2457555.50099.yy.HH.uvcA.first.calfits')[0]
print f

# initialize metrics class
FC = hera_qm.firstcal_metrics.FirstCal_Metrics(f)

# inspect delays
fig = FC.plot_delays()

# inspect only a few antennas
fig = FC.plot_delays(ants=[96, 72, 80, 43, 53])

# inspect delay offsets and save to file
fig = FC.plot_delays(plot_type='fluctuation', save=True)

# filename is by default calfits filestem + .png
get_ipython().system('ls ./*.png')

FC.run_metrics(std_cut=0.5)

FC.metrics['good_sol']

FC.metrics['bad_ants']

FC.metrics['rot_ants']

# plot standard deviation of delay solutions
fig = FC.plot_stds()

# plot z_scores of delay solutions
fig = FC.plot_zscores()

# plot z_scores of delay solutions averaged over time
fig = FC.plot_zscores(plot_type='time_avg')

get_ipython().run_cell_magic('bash', '', 'firstcal_metrics_run.py --help')



