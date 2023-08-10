# print_function for compatibility with Python 3
from __future__ import print_function

# NumPy for numerical computing
import numpy as np

# Pandas for DataFrames
import pandas as pd
pd.set_option('display.max_columns', 100)

# Matplotlib for visualization
import matplotlib.pyplot as plt

# display plots in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Seaborn for easier visualization
import seaborn as sns

get_ipython().run_line_magic('load_ext', 'watermark')
# display library versions
get_ipython().run_line_magic('watermark', '-a "Elie Kawerk" -v -p numpy,pandas,matplotlib,sklearn,seaborn')

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
df.hist(figsize=(10,10), xrot=-45, edgecolor='black')

# Clear the text "residue"
plt.show()

# Summarize numerical features
df.describe()

# Summarize categorical features
df.describe(include=['object'])

# Plot bar plot for each categorical feature
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

for cat_feat in categorical_features:
    sns.countplot(y = cat_feat, data=df)
    plt.show()

# Segment satisfaction by status and plot distributions
sns.violinplot(y='status', x = 'satisfaction', data=df)
plt.show()

# Segment last_evaluation by status and plot distributions
sns.violinplot(y='status', x='last_evaluation',data=df)
plt.show()

# Segment by status and display the means within each class
df.groupby('status').mean()

# Scatterplot of satisfaction vs. last_evaluation
sns.lmplot(y='last_evaluation', x = 'satisfaction', data=df, hue='status', fit_reg=False)
plt.show()

# Scatterplot of satisfaction vs. last_evaluation, only those who have left
sns.lmplot(y='last_evaluation', x = 'satisfaction', data=df[df.status=='Left'], fit_reg=False)
plt.show()

