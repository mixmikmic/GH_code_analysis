# print_function for compatibility with Python 3
from __future__ import print_function

# NumPy for numerical computing
import numpy as np

# Pandas for DataFrames
import pandas as pd

# Matplotlib for visualization
import matplotlib.pyplot as plt

# display plots in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Seaborn for easier visualization
import seaborn as sns

# Scikit-Learn's make_pipeline function
from sklearn.pipeline import make_pipeline

# Scikit-Learn's StandardScaler
from sklearn.preprocessing import StandardScaler

# Scikit-Learn's KMeans algorithm (added later)
from sklearn.cluster import KMeans

# Import analytical base table
base_df = pd.read_csv('analytical_base_table.csv', index_col=0)

# Import thresholded item features
threshold_item_data = pd.read_csv('threshold_item_data.csv', index_col=0)

# Import PCA item features
pca_item_data = pd.read_csv('pca_item_data.csv', index_col=0)

# Print shape of each dataframe
print(base_df.shape)
print(threshold_item_data.shape)
print(pca_item_data.shape)

# Join base_df with threshold_item_data
threshold_df = base_df.join(threshold_item_data)

# Display first 5 rows of threshold_df
threshold_df.head(5)

# Join base_df with pca_item_data
pca_df = base_df.join(pca_item_data)

# Display first 5 rows of pca_df
pca_df.head(5)

# First 5 observations of base_df
base_df.head(5)

# K-Means model pipeline
kmeans = make_pipeline(StandardScaler(), KMeans(n_clusters=3, random_state=123))

# Fit K-Means pipeline
kmeans.fit(base_df)

# Save clusters to base_df
base_df['cluster'] = kmeans.predict(base_df)

# Display first 5 rows of base_df
base_df.head(5)

# Scatterplot, colored by cluster
sns.lmplot(x='total_sales',y = 'avg_cart_value', data=base_df, hue='cluster', fit_reg=False)
plt.show()

# K-Means model pipeline
kmeans = make_pipeline(StandardScaler(), KMeans(n_clusters=3, random_state=123))

# Fit K-Means pipeline
kmeans.fit(threshold_df)

# Save clusters to threshold_df
threshold_df['cluster'] = kmeans.predict(threshold_df)

# Display first 5 rows of threshold_df
threshold_df.head(5)

# Scatterplot, colored by cluster
sns.lmplot(x='total_sales',y = 'avg_cart_value', data=threshold_df, hue='cluster', fit_reg=False)
plt.show()

# K-Means model pipeline
kmeans = make_pipeline(StandardScaler(), KMeans(n_clusters=3, random_state=123))

# Fit K-Means pipeline
kmeans.fit(pca_df)

# Save clusters to pca_df
pca_df['cluster'] = kmeans.predict(pca_df)

# Display first 5 rows of pca_df
pca_df.head(5)

# Scatterplot, colored by cluster
sns.lmplot(x='total_sales',y = 'avg_cart_value', data=pca_df, hue='cluster', fit_reg=False)
plt.show()

# Check all indices are identical
print(all(base_df.index == threshold_df.index))
print(all(base_df.index == pca_df.index))

# Adjusted Rand index
from sklearn.metrics import adjusted_rand_score

# Similary between base_df.cluster and threshold_df.cluster
adjusted_rand_score(base_df.cluster, threshold_df.cluster)

# Similary between threshold_df.cluster and base_df.cluster
adjusted_rand_score(threshold_df.cluster, base_df.cluster)

# Similary between base_df.cluster and pca_df.cluster
adjusted_rand_score(base_df.cluster, pca_df.cluster)

