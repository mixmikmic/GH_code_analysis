# print_function for compatibility with Python 3
from __future__ import print_function
print('Print function is ready to serve!')
# NumPy for numerical computing
import numpy as np

# Pandas for DataFrames
import pandas as pd 
pd.set_option('display.max_columns', 100)

# Matplotlib for visualization
from matplotlib import pyplot as plt

# display plots in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Seaborn for easier visualization
import seaborn as sns

# Load employee data from CSV
df = pd.read_csv('project_files/employee_data.csv')

# Dataframe dimensions
df.shape

# Column datatypes
df.dtypes

# First 10 rows of data
df.head(10)

# Last 10 rows of data
df.tail(10)

# Plot histogram grid
df.hist(xrot=-45, figsize=(10, 10))
# Clear the text "residue"
plt.show()

# Summarize numerical features
df.describe()

# Summarize categorical features
df.describe(include=["object"])

# Plot bar plot for each categorical feature
for features in df.dtypes[df.dtypes == 'object'].index: 
    sns.countplot(y=features, data=df)
    plt.show()

# Segment satisfaction by status and plot distributions
sns.violinplot(y="status", x="satisfaction", data=df)

# Segment last_evaluation by status and plot distributions
sns.violinplot(y="status", x="last_evaluation", data=df)

# Segment by status and display the means within each class
df.groupby('status').mean()

# Scatterplot of satisfaction vs. last_evaluation
sns.lmplot(y="satisfaction", x="last_evaluation", data=df, hue='status', fit_reg=None)

# Scatterplot of satisfaction vs. last_evaluation, only those who have left
sns.lmplot(y="satisfaction", x="last_evaluation", data=df[df.status == 'Left'], hue='status', fit_reg=None)

