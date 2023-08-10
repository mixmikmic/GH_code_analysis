import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from wave_2d_fd_perf import propagators, test_wave_2d_fd_perf, run_timing, run_timing_blocksize
from wave_2d_fd_perf.propagators import (VC1_O2_gcc, VC1_O3_gcc, VC1_Ofast_gcc, VC2_O2_gcc, VC2_O3_gcc, VC2_Ofast_gcc, VC3_Ofast_gcc, VC3_Ofast_unroll_gcc, VC4_Ofast_gcc, VC4_Ofast_extra1_gcc, VC4_Ofast_extra2_gcc, VC4_Ofast_extra3_gcc, VC5_Ofast_gcc, VC6_Ofast_gcc, VC6_256_Ofast_gcc, VC7_Ofast_gcc, VC8_Ofast_gcc, VC9_Ofast_gcc, VC10_Ofast_gcc, VC11_Ofast_gcc, VC12_Ofast_gcc, VC13_Ofast_gcc, VC14_Ofast_gcc, VC15_Ofast_gcc, VF1_O2_gcc, VF1_O3_gcc, VF1_Ofast_gcc, VF2_Ofast_gcc, VF3_Ofast_gcc, VF4_Ofast_gcc, VF5_Ofast_gcc, VF6_Ofast_gcc, VF6_Ofast_autopar_gcc)
from wave_2d_fd_perf.propagators import (VC8a_Ofast_gcc, VC9a_Ofast_gcc, VC10a_Ofast_gcc)

get_ipython().system('lscpu')

get_ipython().system('gcc --version')

blocksizes_y = [1, 8, 16, 32, 64]
blocksizes_x = [8, 16, 32, 64, 128, 256, 512]
t8 = run_timing_blocksize.run_timing_model_size(num_repeat=5, num_steps=10, model_sizes=range(2000, 2500, 5), versions=[{'class': VC8a_Ofast_gcc, 'name': '8'}], blocksizes_y=blocksizes_y, blocksizes_x=blocksizes_x, align=256)

ax = plt.subplot(111)
ax.imshow(t8.groupby(['blocksize_y', 'blocksize_x']).mean()['time'].values.reshape(len(blocksizes_y), len(blocksizes_x)))
plt.xlabel('blocksize_x')
plt.ylabel('blocksize_y')
ax.set_xticks(range(0,len(blocksizes_x)))
ax.set_yticks(range(0,len(blocksizes_y)))
ax.set_yticklabels(blocksizes_y);
ax.set_xticklabels(blocksizes_x);



blocksizes_y = [1, 8]
blocksizes_x = [1024, 2048]
t82 = run_timing_blocksize.run_timing_model_size(num_repeat=1, num_steps=10, model_sizes=range(10000, 10500, 50), versions=[{'class': VC8a_Ofast_gcc, 'name': '8'}], blocksizes_y=blocksizes_y, blocksizes_x=blocksizes_x, align=256)

t82.groupby(['blocksize_y', 'blocksize_x']).mean()

t = run_timing.run_timing_model_size(num_repeat=30, num_steps=10, model_sizes=[2000])

t.plot.bar(x='version', y='time')
plt.ylabel('run time (s)')

t_align256 = run_timing.run_timing_model_size(num_repeat=30, num_steps=10, model_sizes=[2000], align=256)

t_diff = t_align256.copy()
t_diff = t_diff[t_diff['version'] != 'C v6 256 (gcc, -Ofast)'].reset_index(True)
t_diff['diff'] = t_diff['time'] - t['time']
t_diff['perc_diff'] = t_diff['diff'] / t['time'] * 100
t_diff.plot.bar(y='perc_diff', x='version', rot=90)

versions=[{'class': VC6_Ofast_gcc, 'name': 'C v6 (gcc, -Ofast)'}]
t_align = pd.DataFrame()
tmp = run_timing.run_timing_model_size(num_repeat=1, num_steps=10, model_sizes=np.linspace(2000, 2500, 100, dtype=np.int), versions=versions)
tmp['align'] = 0
t_align = t_align.append(tmp, ignore_index=True)
for align in [8, 16, 32, 64, 128, 256, 512, 1024]:
    tmp = run_timing.run_timing_model_size(num_repeat=1, num_steps=10, model_sizes=np.linspace(2000, 2500, 100, dtype=np.int), versions=versions, align=align)
    tmp['align'] = align
    t_align = t_align.append(tmp, ignore_index=True)

t_align.groupby('align').mean().plot(y='time', style='o-')
plt.ylabel('run time (s)');

print('C v6 (gcc, -Ofast)', t_align256.loc[t_align256['version'] == 'C v6 (gcc, -Ofast)', 'time'])
print('C v6 256 (gcc, -Ofast)', t_align256.loc[t_align256['version'] == 'C v6 256 (gcc, -Ofast)', 'time'])

versions=[{'class': VC1_Ofast_gcc, 'name': 'C v1 (gcc, -Ofast)'},
          {'class': VC2_Ofast_gcc, 'name': 'C v2 (gcc, -Ofast)'},
          {'class': VC6_Ofast_gcc, 'name': 'C v6 (gcc, -Ofast)'},
          {'class': VC8_Ofast_gcc, 'name': 'C v8 (gcc, -Ofast)'}]
t_modelsize=run_timing.run_timing_model_size(num_repeat=30, num_steps=10, model_sizes=np.linspace(1000, 5000, 5, dtype=np.int), versions=versions, align=256)

ax=plt.subplot(111)
for v in versions:
    t_modelsize.loc[t_modelsize['version'] == v['name']].plot(x='model_size', y='time', label=v['name'], ax=ax)
plt.ylabel('run time (s)');

t_cellpersec = t_modelsize[t_modelsize['version'] == 'C v6 (gcc, -Ofast)'].copy()
t_cellpersec['num_cells'] = t_cellpersec['model_size']**2
t_cellpersec['cells_per_sec'] = t_cellpersec['num_cells'] / t_cellpersec['time']
t_cellpersec.plot(x='num_cells', y='cells_per_sec')
plt.ylabel('cells per second');

versions=[{'class': VC6_Ofast_gcc, 'name': 'C v6 (gcc, -Ofast)'},
          {'class': VC11_Ofast_gcc, 'name': 'C v11 (gcc, -Ofast)'}]
t_numsteps=run_timing.run_timing_num_steps(num_repeat=100, num_steps=np.linspace(1, 100, 5, dtype=np.int), model_size=2000, versions=versions, align=256)

ax=plt.subplot(111)
for v in versions:
    t_numsteps.loc[t_numsteps['version'] == v['name']].plot(x='num_steps', y='time', label=v['name'], ax=ax)
plt.ylabel('run time (s)')

