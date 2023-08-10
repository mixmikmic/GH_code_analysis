get_ipython().magic("config InlineBackend.figure_format='retina'")
get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams["font.size"] = 14

from sklearn.datasets.california_housing import fetch_california_housing

cal_housing = fetch_california_housing()

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(cal_housing.data,
                                                    cal_housing.target,
                                                    test_size=0.1,
                                                    random_state=1)

names = cal_housing.feature_names

df = pd.DataFrame(data=X_train, columns=names)
df['LogMedHouseVal'] = y_train
_ = df.hist(column=['Latitude', 'Longitude', 'MedInc', 'LogMedHouseVal'])

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence

from mpl_toolkits.mplot3d import Axes3D

clf = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                learning_rate=0.1, loss='huber',
                                random_state=1)
clf.fit(X_train, y_train)

features = [0, 5, 1, 2, (5, 1)]
fig, axs = plot_partial_dependence(clf, X_train, features,
                                   feature_names=names,
                                   n_jobs=3, grid_resolution=50)
fig.suptitle('Partial dependence of house value on nonlocation features\n'
             'for the California housing dataset')
plt.subplots_adjust(top=0.9)  # tight_layout causes overlap with suptitle


fig = plt.figure()

target_feature = (1, 5)
pdp, (x_axis, y_axis) = partial_dependence(clf, target_feature,
                                           X=X_train, grid_resolution=50)
XX, YY = np.meshgrid(x_axis, y_axis)
Z = pdp.T.reshape(XX.shape).T
ax = Axes3D(fig)
surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=plt.cm.BuPu)
ax.set_xlabel(names[target_feature[0]])
ax.set_ylabel(names[target_feature[1]])
ax.set_zlabel('Partial dependence')
#  pretty init view
ax.view_init(elev=22, azim=122)
plt.colorbar(surf)
plt.suptitle('Partial dependence of house value on median age and '
             'average occupancy')
plt.subplots_adjust(top=0.9)

plt.show()

for n, name in enumerate(names):
    print(n, name)

clf = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                learning_rate=0.1, loss='huber',
                                random_state=1)
clf.fit(X_train, y_train)
features = [(7, 6)]


pdp, (x_axis, y_axis) = partial_dependence(clf, features,
                                           X=X_train, grid_resolution=200)

pdp.shape, X_train.shape, x_axis.shape

x_axis.min(), x_axis.max(), y_axis.min(), y_axis.max()

from mpl_toolkits.basemap import Basemap

def california_map(ax=None,
                   lllat=32, urlat=42.5,
                   lllon=-124, urlon=-114):
    m = Basemap(ax=ax, projection='stere',
                lon_0=(urlon + lllon) / 2,
                lat_0=(urlat + lllat) / 2,
                llcrnrlat=lllat, urcrnrlat=urlat,
                llcrnrlon=lllon, urcrnrlon=urlon,
                resolution='l'
               )
    m.drawstates()
    m.drawcountries()
    m.drawcoastlines(color='blue')
    
    m.drawparallels(np.arange(30,50,5),labels=[1,1,0,0])
    m.drawmeridians(np.arange(-130,-110,2),labels=[0,0,0,1])
    
    return m

m = california_map()

XX, YY = np.meshgrid(x_axis, y_axis)
print(XX[0,:10])
print(YY[0,:10])

Z = np.ones_like(XX)
Z = pdp.reshape(XX.shape)

#XX, YY = np.meshgrid(*m(y_axis, x_axis))
#Z = pdp.T.reshape(XX.shape).T

m.contourf(XX, YY, Z, 15, latlon=True)
#m.scatter([-122.47000122, -122.4429158], [32.81999969, 32.81999969], latlon=True)

