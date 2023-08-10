import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from preprocess import base
from visual_data import plot_statistics

get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from preprocess import base
from visual_data import plot_statistics

get_ipython().run_line_magic('matplotlib', 'inline')

root_dir = os.getcwd()
root_dir = root_dir + "/data_0134/smooth_mean_interpolate_bin_mean/"

filelist = base.get_files_csv(root_dir)
len(filelist)
print(filelist)

data = pd.read_csv(root_dir+filelist[0])
plot_statistics.plot_ZX_HW(data)

data = pd.read_csv(root_dir+filelist[1])
plot_statistics.plot_ZX_HW(data)

data = pd.read_csv(root_dir+filelist[2])
plot_statistics.plot_ZX_HW(data)

data = pd.read_csv(root_dir+filelist[3])
plot_statistics.plot_ZX_HW(data)

data = pd.read_csv(root_dir+filelist[4])
plot_statistics.plot_ZX_HW(data)

data = pd.read_csv(root_dir+filelist[5])
plot_statistics.plot_ZX_HW(data)

data = pd.read_csv(root_dir+filelist[6])
plot_statistics.plot_ZX_HW(data)

data = pd.read_csv(root_dir+filelist[7])
plot_statistics.plot_ZX_HW(data)

data = pd.read_csv(root_dir+filelist[8])
plot_statistics.plot_ZX_HW(data)

data = pd.read_csv(root_dir+filelist[9])
plot_statistics.plot_ZX_HW(data)

data = pd.read_csv(root_dir+filelist[10])
plot_statistics.plot_ZX_HW(data)

plot_statistics.plot_stats_ZX_HW_batch(root_dir, np.mean)

plot_statistics.plot_stats_ZX_HW_batch(root_dir, np.std)

print(filelist[0])
data = pd.read_csv(root_dir+filelist[0])
plot_statistics.plot_stats_ZX_WD_No1(data)

print(filelist[1])
data = pd.read_csv(root_dir+filelist[1])
plot_statistics.plot_stats_ZX_WD_No1(data)

print(filelist[2])
data = pd.read_csv(root_dir+filelist[2])
plot_statistics.plot_stats_ZX_WD_No1(data)

print(filelist[3])
data = pd.read_csv(root_dir+filelist[3])
plot_statistics.plot_stats_ZX_WD_No1(data)

print(filelist[4])
data = pd.read_csv(root_dir+filelist[4])
plot_statistics.plot_stats_ZX_WD_No1(data)

print(filelist[5])
data = pd.read_csv(root_dir+filelist[5])
plot_statistics.plot_stats_ZX_WD_No1(data)

print(filelist[6])
data = pd.read_csv(root_dir+filelist[6])
plot_statistics.plot_stats_ZX_WD_No1(data)

print(filelist[7])
data = pd.read_csv(root_dir+filelist[7])
plot_statistics.plot_stats_ZX_WD_No1(data)

print(filelist[8])
data = pd.read_csv(root_dir+filelist[8])
plot_statistics.plot_stats_ZX_WD_No1(data)

print(filelist[9])
data = pd.read_csv(root_dir+filelist[9])
plot_statistics.plot_stats_ZX_WD_No1(data)

print(filelist[10])
data = pd.read_csv(root_dir+filelist[10])
plot_statistics.plot_stats_ZX_WD_No1(data)



