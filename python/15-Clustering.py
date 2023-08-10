# Imports
get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from scipy.cluster.vq import whiten

# Load the iris data
iris = datasets.load_iris()

# Check out the available features
print('\n'.join(iris.feature_names))

# Check out the species ('clusters')
print('\n'.join(iris.target_names))

# The actual data is stored in iris.data
# Let's check how much data there is
[n_samples, n_features] = np.shape(iris.data) 
print("There are ", n_samples , " samples of data, each with " , n_features, " features.")

# Let's set up some indexes, so we know what data we're using
sl_ind = 0    # Sepal Length
sw_ind = 1    # Septal Width
pl_ind = 2    # Petal Length
pw_ind = 3    # Petal Width

# Let's start looking at some data. 
# Let's start with a scatter plot of petal length vs. petal width
fig = plt.figure(1)
plt.scatter(iris.data[:, pl_ind], iris.data[:, pw_ind])

# Add title and labels
plt.title('Iris Data: Petal Length vs. Width', fontsize=16, fontweight='bold')
plt.xlabel('Petal Length', fontsize=14);
plt.ylabel('Petal Width', fontsize=14);

# Plot the data colour coded by species
fig = plt.figure(1)
plt.scatter(iris.data[:, pl_ind][iris.target==0], iris.data[:, pw_ind][iris.target==0],
            c='green', label=iris.target_names[0])
plt.scatter(iris.data[:, pl_ind][iris.target==1], iris.data[:, pw_ind][iris.target==1],
            c='red', label=iris.target_names[1])
plt.scatter(iris.data[:, pl_ind][iris.target==2], iris.data[:, pw_ind][iris.target==2],
            c='blue', label=iris.target_names[2])

# Add title, labels and legend
plt.title('Iris Data: Petal Length vs. Width', fontsize=16, fontweight='bold')
plt.xlabel('Petal Length', fontsize=14);
plt.ylabel('Petal Width', fontsize=14);
plt.legend(scatterpoints=1, loc='upper left');

# Note that splitting up the plotting per group is basically a hack to make the legend work, 
# The following command plots the data perfectly well, colour coded by target:
#  plt.scatter(iris.data[:, petal_length_ind], iris.data[:, petal_width_ind], c=iris.target)
# However, it's a pain to get a labelled legend when plotted this way

# Pull out the data of interest - Petal Length & Petal Width
d1 = np.array(iris.data[:, pl_ind])
d2 = np.array(iris.data[:, pw_ind])

# Check out the whiten function
get_ipython().magic('pinfo whiten')

# Whiten Data
d1w = whiten(d1)
d2w = whiten(d2)

# Combine data into shape for skl
data = np.vstack([d1w, d2w]).T

# Initialize KMeans object, set to fit 3 clusters
km = KMeans(n_clusters=3, random_state=13)

# Fit the data with KMeans
km.fit(data)

# Let's check out the clusters that KMeans found
plt.scatter(d1, d2, c=km.labels_);
plt.xlabel('Year');
plt.ylabel('Age');

# Add title, labels and legend
plt.title('Iris Data: PL vs. PW Clustered', fontsize=16, fontweight='bold')
plt.xlabel('Petal Length', fontsize=14);
plt.ylabel('Petal Width', fontsize=14);

