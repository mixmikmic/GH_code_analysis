import data_tools as dt
import data_dir as dd
import matplotlib.pyplot as plt
import numpy as np

data = dt.import_nasa_dataset(dd.PHM_TRAIN)

plt.plot(data[:223, 5:])
plt.show()

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

kmeans = KMeans(n_clusters=6).fit(data[:, 2:5])

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.scatter(data[:223, 2], data[:223, 3], data[:223, 4])
ax.plot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], 'rx')

plt.show()

clusters = kmeans.predict(data[:, 2:5]).reshape(-1, 1)

regimes = []
for i in xrange(6):
    regimes.append(np.where(clusters == i)[0])

test_data = np.copy(data[:, 5:])
for i in xrange(len(regimes)):
    mean = test_data[regimes[i]].mean(axis=0)
    std = test_data[regimes[i]].std(axis=0)
    test_data[regimes[i]] = np.divide(np.subtract(test_data[regimes[i]], mean), std)

min = test_data[:, [1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 19]].min()
max = test_data[:, [1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 19]].max()

#test_data[:, [1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 19]] = np.divide(np.subtract(
#test_data[:, [1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 19]], min), max - min)

plt.plot(test_data[:223, [1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 19]])
plt.show()

from sklearn import preprocessing
plt.plot(preprocessing.scale(test_data[:223, [1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 19]]))
plt.show()

plt.plot(preprocessing.minmax_scale(test_data[:223, [1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 19]], (-1, 1)))
plt.show()

for i in xrange(test_data.shape[1]):
    print "SENSOR ", i + 1, ":"
    plt.plot(test_data[:223, i])
    plt.show()

from sklearn import preprocessing
plt.plot(test_data[:223, [1, 2, 6, 7, 8, 10, 11, 12, 13, 14, 19]])
plt.show()

regime_phm_data = dt.get_regime_partitioned_units(dd.PHM_TRAIN)

minimum = regime_phm_data[0].min()
maximum = regime_phm_data[0].max()
for unit in regime_phm_data:
    if minimum > unit.min():
        minimum = unit.min()
    if maximum < unit.max():
        maximum = unit.max()

print min, max

sensors = [6, 7, 8, 11, 12, 13, 15, 16, 17, 18, 19, 24]

print "CMAPSS 1"
plt.plot(preprocessing.minmax_scale(dt.get_nasa_units(dd.CMAPSS_TRAIN_1)[0][:, 5:], (-1, 1)))
plt.show()
plt.plot(dt.get_nasa_units(dd.CMAPSS_TEST_1)[0][:, 5:])
plt.show()

print "CMAPSS 2"
plt.plot(dt.get_regime_partitioned_units(dd.CMAPSS_TRAIN_2)[0][:, 5:], 'x')
plt.show()
plt.plot(dt.get_regime_partitioned_units(dd.CMAPSS_TEST_2)[1][:, 5:], 'x')
plt.show()

print "CMAPSS 3"
plt.plot(dt.get_nasa_units(dd.CMAPSS_TRAIN_3)[0][:, :], 'x')
plt.show()
plt.plot(dt.get_nasa_units(dd.CMAPSS_TEST_3)[1][:, :], 'x')
plt.show()

print "CMAPSS 4"
plt.plot(dt.get_regime_partitioned_units(dd.CMAPSS_TRAIN_4)[0][:, 5:], 'x')
plt.show()
plt.plot(dt.get_regime_partitioned_units(dd.CMAPSS_TEST_4)[1][:, 5:], 'x')
plt.show()

print "PHM"
indices = np.where(data[:, 0] == 1)[0]
plt.plot(test_data[indices[0]:indices[len(indices) - 1]+1, :], 'x')
plt.show()
plt.plot(dt.get_regime_partitioned_units(dd.PHM_TRAIN)[0][:55, 5:], 'x')
plt.show()
print "PHM TEST[0]"
plt.plot(dt.get_regime_partitioned_units(dd.PHM_TEST)[0][:, 5:], 'x')
plt.show()
plt.plot(preprocessing.scale(dt.get_regime_partitioned_units(dd.PHM_TEST)[0][:, 5:]), 'x')
plt.show()
plt.plot(preprocessing.minmax_scale(dt.get_regime_partitioned_units(dd.PHM_TEST)[0][:, 5:], (-1, 1)), 'x')
plt.show()
plt.plot(np.divide(np.subtract(preprocessing.scale(dt.get_regime_partitioned_units(dd.PHM_TEST)[0][:, 5:]), minimum), maximum - minimum), 'x')
plt.show()

clusters = kmeans.predict(dt.import_nasa_dataset(dd.PHM_TEST)[:, 2:5]).reshape(-1, 1)

regimes = []
for i in xrange(6):
    regimes.append(np.where(clusters == i)[0])

# test_data_2 = np.copy(data[:, 5:])
test_data_2 = dt.import_nasa_dataset(dd.PHM_TEST)[:, 5:]
for i in xrange(len(regimes)):
    minimum = np.amin(test_data[regimes[i]], axis=0)
    maximum = np.amax(test_data[regimes[i]], axis=0)
    test_data_2[regimes[i]] = np.subtract(np.multiply(2, np.divide(np.subtract(test_data[regimes[i]], minimum), 
                                                         np.subtract(maximum, minimum))), 1)

print test_data_2.shape
plt.plot(test_data_2[:223, [1, 2, 3, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 19]])
plt.show()



