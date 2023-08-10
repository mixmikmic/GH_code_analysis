import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
matplotlib.style.use('ggplot')
get_ipython().magic('matplotlib inline')

#times=wave_1d_fd_perf.run_timing.run_timing_num_steps()
#times.to_csv('times_num_steps.csv')
times = pd.read_csv('times_num_steps.csv')

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_yscale('log')
ax.set_ylabel('time (s)')
times.groupby('version').plot(x='num_steps', y='time', ax=ax, style='o-', legend=False)

times100 = times.loc[times['num_steps'] == 100].copy()
times100['speedup']=(times100.loc[times100['version']=='Python v1']['time'].values)/times100['time']

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_ylabel('speedup (times)')
times100.plot(kind='bar', x='version', y='speedup', logy=True, legend=False, ax=ax)

#times=wave_1d_fd_perf.run_timing.run_timing_model_size()
#times.to_csv('times_model_size.csv')
times = pd.read_csv('times_model_size.csv')

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_yscale('log')
ax.set_ylabel('time (s)')
times.groupby('version').plot(x='model_size', y='time', ax=ax, style='o-', legend=False)

times2000 = times.loc[times['model_size'] == 2000].copy()
times2000['speedup']=(times2000.loc[times2000['version']=='Python v1']['time'].values)/times2000['time']

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_ylabel('speedup (times)')
times2000.plot(kind='bar', x='version', y='speedup', logy=True, legend=False, ax=ax)

#times=wave_1d_fd_perf.run_timing.run_timing_model_size(num_steps=2000, model_sizes=[2000])
#times.to_csv('times_2000_2000.csv')
times=pd.read_csv('times_2000_2000.csv')

times['speedup']=(times.loc[times['version']=='Python v1']['time'].values)/times['time']
times['speedup']

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_ylabel('speedup (times)')
times.plot(kind='bar', x='version', y='speedup', logy=True, legend=False, ax=ax)

