import numpy as np
import pandas as pd

rankings_df = pd.read_csv('massey_rankings_compare.csv',header=1)
# delete the last extra blank column
rankings_df = rankings_df.iloc[:, :-1]
rankings_df.head()

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('classic')

# use pandas to plot scatter matrix, just pick out a small sample of rankings
from pandas.tools.plotting import scatter_matrix
models = [" YUSAG", " DOK", " MAS", " DII", " PAY", " TPR"]
scatter_matrix(rankings_df[models], alpha=0.2, figsize=(8,6))
plt.show()

# get rid of team names
rankings_df = rankings_df.iloc[:,1:]
# make the matrix
corr_matrix = rankings_df.corr()

corr_matrix[" YUSAG"].sort_values(ascending=False)

import seaborn as sns
sns.heatmap(corr_matrix,  
            cmap="YlGnBu",
            xticklabels=corr_matrix.columns.values,
            yticklabels=corr_matrix.columns.values)

import seaborn as sns
sns.heatmap(corr_matrix,  
            center = 0.84,
            xticklabels=corr_matrix.columns.values,
            yticklabels=corr_matrix.columns.values)

