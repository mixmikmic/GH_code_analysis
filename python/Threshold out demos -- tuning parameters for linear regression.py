import matplotlib.pyplot as plt
get_ipython().magic('matplotlib nbagg')

import thresholdOut_mydemo_paramTuning as my_demo
#from importlib import reload
#reload(my_demo)

# number of permutations in the experiment
reps = 50

# number of samples per data block, and data dimension
n, d = 50, 500

# number of steps to optimize over in the grid search
grid_size = 100

# thresholdout 'scale' factor
tho_scale = 0.5

f_cls, ax_cls = my_demo.runExpt_and_makePlots(n, d, grid_size, reps, tho_scale=tho_scale, is_classification=True)

f_cls_bigscale = my_demo.runExpt_and_makePlots(n, d, grid_size, reps, tho_scale=100.0, is_classification=True)

f_cls_smlscale = my_demo.runExpt_and_makePlots(n, d, grid_size, reps, tho_scale=0.00001, is_classification=True)

reps = 50
n, d = 50, 500
grid_size = 100

tho_scale = 0.5

f_cls, ax_cls = my_demo.runExpt_and_makePlots(n, d, grid_size, reps, tho_scale=tho_scale, is_classification=False)



