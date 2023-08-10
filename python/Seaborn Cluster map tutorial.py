import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# laod flight data set
iris_data = sns.load_dataset("iris")
iris_data.head()

# Plot a clustered heatmap:
species = iris_data.pop("species")
sns.clustermap(iris_data)

## Use a different similarity metric:
sns.clustermap(iris_data,metric = 'correlation')

# Use a different clustering method:
sns.clustermap(iris_data,method = 'single')

# Use a different colormap and ignore outliers in colormap limits:

sns.clustermap(iris_data, cmap="mako", robust=True)

# Change the size of the figure:
sns.clustermap(iris_data, figsize=(6, 7))

# Plot one of the axes in its original organization:

sns.clustermap(iris_data, col_cluster=False)

# Add colored labels:

lut = dict(zip(species.unique(), "rbg"))
row_colors = species.map(lut)
g = sns.clustermap(iris_data, row_colors=row_colors)

# Standardize the data within the columns:

sns.clustermap(iris_data, standard_scale=1)

# Normalize the data within the rows:

sns.clustermap(iris_data, z_score=0)



