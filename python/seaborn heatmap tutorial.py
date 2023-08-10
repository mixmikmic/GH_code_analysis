import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

# Plot a heatmap for a numpy array:

#uniform_data = np.random.rand(10,12)
#uniform_data = np.arange(1,17).reshape(4,4)
sns.heatmap(uniform_data)



x = np.array([[1,2,3,4],[2,3,4,1],[5,4,2,1],[6,7,8,5]])
sns.heatmap(x)
x

df = pd.DataFrame(np.random.random((5,5)), columns=["a","b","c","d","e"])

# Default heatmap: just a visualization of this square matrix
p1 = sns.heatmap(df)



