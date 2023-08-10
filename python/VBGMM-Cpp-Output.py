get_ipython().magic('matplotlib inline')

import os
import matplotlib.pyplot as plt
import pandas as pd

os.chdir("/home/3928941380/Desktop/Files/C_Practice/")

res = pd.read_csv("VBGMM-output.txt")

res[res["parameter"]=="alpha"] # 混合係数の期待値

res[res["parameter"]=="nk"]

res[res["parameter"]=="mu"]

# select use class
use_class = list(res[res["parameter"]=="alpha"].ix[list(res.ix[res["parameter"]=="alpha", "value1"] > 0.1), "class"])

# Plot of mu_k

fp = open("faithful.txt")
data_x = []
data_y = []
for row in fp:
    data_x.append(float((row.split()[0])))
    data_y.append(float((row.split()[1])))
fp.close()
plt.plot(data_x, data_y, "wo", markersize=5)

for i in use_class:
    x1 = res.ix[(res["parameter"]=="mu")&(res["class"] == i), "value1"]
    x2 = res.ix[(res["parameter"]=="mu")&(res["class"] == i), "value2"]
    
    plt.plot(x1, x2, "*", markersize=12, color="b")

use_class = [0,1,2,3,4,5]

# Plot of mu_k

fp = open("faithful.txt")
data_x = []
data_y = []
for row in fp:
    data_x.append(float((row.split()[0])))
    data_y.append(float((row.split()[1])))
fp.close()
plt.plot(data_x, data_y, "wo", markersize=5)

for i in use_class:
    x1 = res.ix[(res["parameter"]=="mu")&(res["class"] == i), "value1"]
    x2 = res.ix[(res["parameter"]=="mu")&(res["class"] == i), "value2"]
    
    plt.plot(x1, x2, "*", markersize=12, color="b")

