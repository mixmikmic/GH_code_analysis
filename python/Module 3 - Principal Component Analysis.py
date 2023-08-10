# print_function for compatibility with Python 3
from __future__ import print_function

# NumPy for numerical computing
import numpy as np
# Pandas for DataFrames
import pandas as pd


# Matplotlib for visualization
from matplotlib import pyplot as plt

# display plots in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Seaborn for easier visualization
import seaborn as sns

# StandardScaler from Scikit-Learn
from sklearn.preprocessing import StandardScaler

# PCA from Scikit-Learn (added later)
from sklearn.decomposition import PCA

# Read item_data.csv
item_data = pd.read_csv('item_data.csv', index_col=0)

# Display item_data's shape
item_data.shape

# Set random seed
np.random.seed(101)

# Create first feature: x1
x1 = np.random.normal(0, 1, 100)

# Create second feature: x2
x2 = x1 + np.random.normal(0, 1, 100)

# Stack together as columns
X = np.stack([x1, x2], axis=1)

# Print shape of X
X.shape

# Initialize instance of StandardScaler
scaler = StandardScaler()

# Fit and transform X
X_scaled = scaler.fit_transform(X)

# Display first 5 rows of X_scaled
X_scaled[:5]

# Plot scatterplot of scaled x1 against scaled x2
plt.scatter(X_scaled[:,0], X_scaled[:,1])
# Put plot axes on the same scale
plt.axis('equal')


# Label axes
plt.xlabel('x1 (scaled)')
plt.ylabel('x2 (scaled)')
# Clear text residue
plt.show()

# Initialize instance of PCA transformation
pca = PCA()

# Fit the instance
pca.fit(X_scaled)

# Display principal components
pca.components_

# Plot scaled dataset and make it partially transparent
plt.scatter(X_scaled[:,0], X_scaled[:,1], alpha=0.3)

# Plot first principal component in black
plt.plot([0, 2*pca.components_[0,0]], [0, 2*pca.components_[0,1]], 'k')

# Plot second principal component in red
plt.plot([0, pca.components_[1,0]], [0, pca.components_[1,1]], 'r')

# Set axes
plt.axis('equal')
plt.xlabel('x1 (scaled)')
plt.ylabel('y1 (scaled)')


# Clear text residue
plt.show()

# Generate new features
PC = pca.transform(X_scaled)

# Plot transformed dataset
plt.scatter(PC[:,0], PC[:,1], alpha=0.3, color='g')

# Plot first principal component in black
plt.plot([0, 2], [0,0], 'k')

# Plot second principal component in red
plt.plot([0,0], [0,1], 'r')

# Set axes
plt.axis('equal')
plt.xlabel('PC1')
plt.ylabel('PC2')

# Clear text residue
plt.show()

# Display explained variance ratio
pca.explained_variance_ratio_

# Initialize and fit a PCA transformation, only keeping 1 component
pca = PCA(n_components=1)
pca.fit(X_scaled)

# Display principal components
pca.components_

# Generate new features
PC = pca.transform(X_scaled)
print(PC.shape)

# Plot transformed dataset
plt.scatter(PC[:,0], len(PC)*[0], alpha=0.3, color='g')


# Plot first principal component in black
plt.plot([0,2],[0,0], 'k')

# Set axes
plt.axis('equal')
plt.xlabel('PC1')

# Clear text residue
plt.show()

# Initialize instance of StandardScaler
sc = StandardScaler()

# Fit and transform item_data
item_data_scaled = scaler.fit_transform(item_data)
# Display first 5 rows of item_data_scaled
item_data_scaled[:5]

# Initialize and fit a PCA transformation
pca = PCA()
pca.fit(item_data_scaled)

# Generate new features
PC_items = pca.transform(item_data_scaled)

# Display first 5 rows
PC_items[:5]

# Cumulative explained variance
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot cumulative explained variance
plt.plot(range(len(cumulative_explained_variance)), cumulative_explained_variance)

# How much variance we'd capture with the first 125 components
cumulative_explained_variance[124]

# Initialize PCA transformation, only keeping 125 components
pca = PCA(n_components=125)

# Fit and transform item_data_scaled
PC_items = pca.fit_transform(item_data_scaled)

# Display shape of PC_items
PC_items.shape

# Put PC_items into a dataframe
items_pca = pd.DataFrame(PC_items)

# Name the columns
items_pca.columns = ['PC{}'.format(i + 1) for i in range(PC_items.shape[1])]

# Update its index
items_pca.index = item_data.index

# Display first 5 rows
items_pca.head()

# Save pca_item_data.csv
items_pca.to_csv('pca_item_data.csv')

