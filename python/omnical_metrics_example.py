get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import hera_qm as hqm
from hera_qm.data import DATA_PATH
import os

omni_file = os.path.join(DATA_PATH, "zen.2457555.42443.xx.HH.uvcA.reallybad.omni.calfits")
fc_file = os.path.join(DATA_PATH, "zen.2457555.42443.xx.HH.uvcA.reallybad.first.calfits")

OM = hqm.omnical_metrics.OmniCal_Metrics(omni_file)
full_metrics = OM.run_metrics(fcfiles=fc_file, cut_edges=True, phs_std_cut=0.3, chisq_std_zscore_cut=4.0)

# print metrics keys
print "full_metrics keys:"
print full_metrics.keys()
print ""
print "full_metrics['XX'] keys:"
print full_metrics['XX'].keys()
print ""
print "full_metrics['XX']['chisq_tot_avg'] value:"
print full_metrics['XX']['chisq_tot_avg']

# Look at Chi Square Standard Deviations
fig = hqm.omnical_metrics.plot_chisq_metric(full_metrics['XX'])

fig = hqm.omnical_metrics.plot_phs_metric(full_metrics['XX'], plot_type='std')

print full_metrics['XX']['chisq_good_sol']
print full_metrics['XX']['ant_phs_std_good_sol']

# time-average chi-square per antenna
fig = OM.plot_chisq_tavg()

# gain amplitude at a given time
fig = OM.plot_gains(plot_type='amp')

# plot firstcal-subtracted gain phase at a given time
fig = OM.plot_gains(plot_type='phs', divide_fc=True)

fig = hqm.omnical_metrics.plot_phs_metric(full_metrics['XX'], plot_type='hist')

fig = hqm.omnical_metrics.plot_phs_metric(full_metrics['XX'], plot_type='ft')

omni_file = os.path.join(DATA_PATH, "zen.2457555.42443.xx.HH.uvcA.bad.omni.calfits")
fc_file = os.path.join(DATA_PATH, "zen.2457555.42443.xx.HH.uvcA.first.calfits")

OM = hqm.omnical_metrics.OmniCal_Metrics(omni_file)
full_metrics = OM.run_metrics(fcfiles=fc_file, cut_edges=True, phs_std_cut=0.3, chisq_std_zscore_cut=4.0)

# gain amplitude at a given time
fig = OM.plot_gains(plot_type='amp')

# plot firstcal-subtracted gain phase at a given time
fig = OM.plot_gains(plot_type='phs', divide_fc=True)

# Look at Chi Square Standard Deviations
fig = hqm.omnical_metrics.plot_chisq_metric(full_metrics['XX'])

fig = hqm.omnical_metrics.plot_phs_metric(full_metrics['XX'], plot_type='std')

# time-average chi-square per antenna
fig = OM.plot_chisq_tavg()

fig = hqm.omnical_metrics.plot_phs_metric(full_metrics['XX'], plot_type='hist')

fig = hqm.omnical_metrics.plot_phs_metric(full_metrics['XX'], plot_type='ft')

print full_metrics['XX']['chisq_good_sol']
print full_metrics['XX']['ant_phs_std_good_sol']

omni_file = os.path.join(DATA_PATH, "zen.2457555.42443.xx.HH.uvcA.good.omni.calfits")
fc_file = os.path.join(DATA_PATH, "zen.2457555.42443.xx.HH.uvcA.first.calfits")

OM = hqm.omnical_metrics.OmniCal_Metrics(omni_file)
full_metrics = OM.run_metrics(fcfiles=fc_file, cut_edges=True, phs_std_cut=0.3, chisq_std_zscore_cut=4.0)

# gain amplitude at a given time
fig = OM.plot_gains(plot_type='amp')

# plot firstcal-subtracted gain phase at a given time
fig = OM.plot_gains(plot_type='phs', divide_fc=True)

# Look at Chi Square Standard Deviations
fig = hqm.omnical_metrics.plot_chisq_metric(full_metrics['XX'])

fig = hqm.omnical_metrics.plot_phs_metric(full_metrics['XX'], plot_type='std')

# time-average chi-square per antenna
fig = OM.plot_chisq_tavg()

fig = hqm.omnical_metrics.plot_phs_metric(full_metrics['XX'], plot_type='hist')

fig = hqm.omnical_metrics.plot_phs_metric(full_metrics['XX'], plot_type='ft')

print full_metrics['XX']['chisq_good_sol']
print full_metrics['XX']['ant_phs_std_good_sol']

get_ipython().run_cell_magic('bash', '', 'omnical_metrics_run.py -h')



