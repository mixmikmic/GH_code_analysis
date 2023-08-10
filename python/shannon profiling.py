import numpy as np
from shannon.continuous import mi
import timeit
import matplotlib.pyplot as plt
from deepretina.io import despine

get_ipython().magic('matplotlib inline')
#%matplotlib qt

from pylab import rcParams
rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
rcParams['image.interpolation'] = 'nearest'
rcParams['image.cmap'] = 'gray'

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

x1 = np.random.randn(1000,)
x2 = 0.02*x1 + np.random.randn(1000,)

get_ipython().magic("timeit mi(x1, x2, method='ann')")

get_ipython().magic("timeit mi(x1, x2, method='nearest-neighbors')")

mi(x1, x2, method='ann')

mi(x1, x2, method='nearest-neighbors')

lengths = np.logspace(1,3.5,25)

from timeit import timeit

naive_durations = []
sklearn_durations = []
repeats = 10
for l in lengths:
    naive_timers = []
    skl_timers = []
    setup_script = 'from shannon.continuous import mi; import numpy as np; x1 = np.random.randn(%i,); x2 = 0.5*x1 + np.random.randn(%i,)' %(l,l)
    t_naive = timeit("mi(x1, x2, method='nearest-neighbors')", number=repeats, setup=setup_script)
    t_skl = timeit("mi(x1, x2, method='ann')", number=repeats, setup=setup_script)
        
    naive_durations.append(t_naive)
    sklearn_durations.append(t_skl)

# naive_durations = []
# sklearn_durations = []
# repeats = 10
# for l in lengths:
#     naive_timers = []
#     skl_timers = []
#     for r in range(repeats):
#         start_naive = timeit.timeit()
#         tmp = mi(x1, x2, method='nearest-neighbors')
#         end_naive = timeit.timeit()
        
#         start_skl = timeit.timeit()
#         tmp = mi(x1, x2, method='ann')
#         end_skl = timeit.timeit()
        
#         naive_timers.append(end_naive-start_naive)
#         skl_timers.append(end_skl-start_skl)
        
#     naive_durations.append(np.mean(naive_timers))
#     sklearn_durations.append(np.mean(skl_timers))



plt.plot(lengths, naive_durations, 'k.-', linewidth=5, markersize=15, label='naive')
plt.plot(lengths, sklearn_durations, 'r.-', linewidth=5, markersize=15, label='ball_tree')
plt.ylabel('Time (seconds)', fontsize=20)
plt.xlabel('Size of array', fontsize=20)
plt.xscale('log')
plt.legend()
despine(plt.gca())

get_ipython().magic('pinfo plt.legend')

lengths = np.logspace(1,3.5,25).astype('int')

lengths[0]

from timeit import timeit

ball_tree_durations = []
brute_force_durations = []
kd_tree_durations = []
auto_durations = []
repeats = 10
for l in lengths:
    print(l)
    naive_timers = []
    skl_timers = []
    setup_script = 'from shannon.continuous import entropy; import numpy as np; x1 = np.random.randn(%i,1)' %(l)
    t_ball = timeit("entropy(x1, method='ann', algo='ball_tree')", number=repeats, setup=setup_script)
    t_kd_tree = timeit("entropy(x1, method='ann', algo='kd_tree')", number=repeats, setup=setup_script)
    t_auto = timeit("entropy(x1, method='ann', algo='auto')", number=repeats, setup=setup_script)
    t_brute = timeit("entropy(x1, method='ann', algo='brute')", number=repeats, setup=setup_script)
        

    ball_tree_durations.append(t_ball)
    brute_force_durations.append(t_brute)
    kd_tree_durations.append(t_kd_tree)
    auto_durations.append(t_auto)

plt.plot(lengths, auto_durations, 'k.-', linewidth=5, markersize=15, label='auto', alpha=0.5)
plt.plot(lengths, ball_tree_durations, 'r.-', linewidth=5, markersize=15, label='ball_tree', alpha=0.5)
plt.plot(lengths, brute_force_durations, 'c.-', linewidth=5, markersize=15, label='brute', alpha=0.5)
plt.plot(lengths, kd_tree_durations, 'b.-', linewidth=5, markersize=15, label='kd_tree', alpha=0.5)
plt.ylabel('Time (seconds)', fontsize=20)
plt.xlabel('Size of array', fontsize=20)
plt.xscale('log')
plt.yscale('log')
plt.legend()
despine(plt.gca())



