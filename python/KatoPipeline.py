import sys
import os

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
import numpy as np

sys.path.append('../src/')
sys.path.append('../lib/')
import data_config as dc
from tvregdiff import TVRegDiff
from analysis import Analyzer

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload')

wormData = dc.kato_matlab.data()

derivData = Analyzer(wormData[0]['deltaFOverF_deriv'][:,0:500].T)
rawData = Analyzer(wormData[0]['deltaFOverF'][:,0:500].T)
print(derivData.timeseries.shape)
print(rawData.timeseries.shape)

computedDerivData = Analyzer(rawData.tvd(iters=20, alpha=1e-1, diff=None))

fig = plt.figure(figsize=(40,20))

ax_raw1 = fig.add_subplot(2, 3, 1)
ax_deriv1 = fig.add_subplot(2, 3, 2)
ax_pca1 = fig.add_subplot(2, 3, 3, projection='3d')

ax_raw2 = fig.add_subplot(2,3,4)
ax_deriv2 = fig.add_subplot(2,3,5)
ax_pca2 = fig.add_subplot(2,3,6, projection='3d')

ax_raw1.set_title("Kato's raw data")
ax_deriv1.set_title("Kato's computed derivative")
ax_pca1.set_title("OpenWorm PCA on Kato's computed derivative")

ax_raw2.set_title("Kato's raw data - same as above")
ax_deriv2.set_title("OpenWorm computed derivative of Kato's raw data")
ax_pca2.set_title("OpenWorm PCA on OpenWorm computed derivative")

rawData.timeseries_plot(ax_raw1)
derivData.timeseries_plot(ax_deriv1)
derivData.pca_plot3d(ax_pca1)

rawData.timeseries_plot(ax_raw2)
computedDerivData.timeseries_plot(ax_deriv2)
computedDerivData.pca_plot3d(ax_pca2)

plt.show()

