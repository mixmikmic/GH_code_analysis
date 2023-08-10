import numpy as np

import CBECSLib

atlanta_nw = (33.8961, -84.4650)
atlanta_se = (33.6975, -84.2995)

X,Y,columnNames,classVals = CBECSLib.getDataset(1,pbaOneHot=True)
np.save("output/dummyFeatures.npy",X)

points = []
for i in range(X.shape[0]):
    lat, lon = np.random.uniform(atlanta_se[0], atlanta_nw[0]), np.random.uniform(atlanta_nw[1], atlanta_se[1])
    points.append((lat,lon))
points = np.array(points)
np.save("output/dummyFeatureLocations.npy", points)

