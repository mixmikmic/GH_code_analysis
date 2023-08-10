import numpy as np

data = np.random.rand(100, 2)

get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt

x = [item[0] for item in data]
y = [item[1] for item in data]

# x, y = zip(*data)

plt.scatter(x, y)
plt.show()

from sklearn.cluster import KMeans

estimator = KMeans(n_clusters=4)
estimator.fit(data)

colours = ['r', 'g', 'b', 'y']  # red, green, blue, yellow

predicted_colours = [colours[label] for label in estimator.labels_]

plt.scatter(x, y, c=predicted_colours)
plt.show()



