# print_function for compatibility with Python 3
from __future__ import print_function

# NumPy for numerical computing
import numpy as np

# Pandas for DataFrames
import pandas as pd
pd.set_option('display.max_columns',100)

# Matplotlib for visualization
import matplotlib.pyplot as plt

# display plots in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Seaborn for easier visualization
import seaborn as sns

# Load employee data from CSV
df = pd.read_csv('project_files/employee_data.csv')

# Drop duplicates
df.drop_duplicates(inplace=True)
df.shape

# Unique classes of 'department'
df.department.unique()

# Drop temporary workers
df = df[df['department'] != 'temp']
df.shape

# Print unique values of 'filed_complaint'
print(df['filed_complaint'].unique())

# Print unique values of 'recently_promoted'
print(df['recently_promoted'].unique())

# Missing filed_complaint values should be 0
df['filed_complaint'].replace(np.nan, 0, inplace=True)

# Missing recently_promoted values should be 0
df['recently_promoted'].replace(np.nan, 0, inplace= True)

# Print unique values of 'filed_complaint'
print(df['filed_complaint'].unique())

# Print unique values of 'recently_promoted'
print(df['recently_promoted'].unique())

# 'information_technology' should be 'IT'
df['department'].replace('information_technology', 'IT', inplace=True)

# Plot class distributions for 'department'
plt.figure(figsize=(8,8))
sns.countplot(y='department', data=df)
plt.show()

# Display number of missing values by feature
df.isnull().sum()

# Fill missing values in department with 'Missing'
df['department'].fillna('Missing',inplace=True)

# Indicator variable for missing last_evaluation
df['last_evaluation_missing'] = df['last_evaluation'].isnull().astype(int)

# Fill missing values in last_evaluation with 0
df['last_evaluation'].fillna(0, inplace=True)

# Display number of missing values by feature
df.isnull().sum()

# Scatterplot of satisfaction vs. last_evaluation, only those who have left
sns.lmplot(x='satisfaction', y= 'last_evaluation', fit_reg=False,
           data = df.loc[df['status']=='Left'])
plt.show()

# Create indicator features
df['underperformer'] = ((df['last_evaluation'] < 0.6) & (df['last_evaluation_missing'] == 0)).astype(int)

df['unhappy'] = (df['satisfaction'] < 0.2).astype(int)

df['overachiever'] = ((df['last_evaluation'] > 0.8) & (df['satisfaction'] > 0.7)).astype(int)

# The proportion of observations belonging to each group
df[['underperformer', 'unhappy', 'overachiever']].mean()

# Convert status to an indicator variable
df['status'] = pd.get_dummies( df.status ).Left

# The proportion of observations who 'Left'
df.status.mean()

# Create new dataframe with dummy features
categorical_feats = df.select_dtypes(include=['object']).columns.tolist()
df = pd.get_dummies(df, columns= categorical_feats)

# Display first 10 rows
df.head(10)

# Save analytical base table
df.to_csv('analytical_base_table.csv', index= None)

