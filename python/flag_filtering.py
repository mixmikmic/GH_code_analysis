import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocess import base
import os

get_ipython().run_line_magic('matplotlib', 'inline')

def plot_flag(flag_data, file):
    fig, axs = plt.subplots(5,1, sharex=True, figsize=(10, 5))
    axs[0].set_title(file)
    for name, ax in zip(flag_data.columns, axs):
        series = flag_data[name]
        x = np.arange(len(series))
        ax.scatter(x, series)
        ax.set_ylabel(name)

root_dir = os.getcwd()
print(root_dir)

path = root_dir+"/data/pharsed_data/"
filelist =base.get_files_csv(path)
for file in filelist:
    data = pd.read_csv(path+file)
    flag_data = data[['XD_FLAG', 'ZD_FLAG', 'JY_FLAG', 'SP_FLAG', 'ZX_FLAG']]
    print("plot for file:  {}".format(file))
    plot_flag(flag_data, file)

path = root_dir+"/data/coordinated_zd_zx_sp_flag/"
filelist = base.get_files_csv(path)
for file in filelist:
    data = pd.read_csv(path+file)
    flag_data = data[['XD_FLAG', 'ZD_FLAG', 'JY_FLAG', 'SP_FLAG', 'ZX_FLAG']]
    print("plot for file:  {}".format(file))
    plot_flag(flag_data, file)

# 01_233_0143_CC-66666_2016-05-20
data = pd.read_csv("data/coordinated_zd_zx_sp_flag/01_233_0143_CC-66666_2016-05-20.csv")
print(data.info())

s = data.BTSJ[0]
print(s)
s2 = s.split(' ')[1]
s2 = s2.split(':')
s2

print(int(s2[0])*60**2)
print(int(s2[1])*60)
print(int(s2[2]))

s1 = s.split(' ')[0]
print(s1)
dates = s1.split('-')
print(dates[2])

date, time = s.split(' ')
print(date)
print(time)



