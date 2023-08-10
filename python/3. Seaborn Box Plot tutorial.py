import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

tips = sns.load_dataset('tips')
tips.head()

sns.set_style("whitegrid")
sns.boxplot(x = tips['size'])

sns.set_style("whitegrid")
sns.boxplot(x = tips['total_bill'])

# Draw a vertical boxplot grouped by a categorical variable:

sns.set_style("whitegrid")
sns.boxplot(x = 'day',y = 'total_bill',data = tips)

# Draw a boxplot with nested grouping by two categorical variables:

sns.set_style("whitegrid")
sns.boxplot(x = 'day',y = 'total_bill',hue = 'smoker',data = tips,palette = 'Set3')

sns.set_style("whitegrid")
sns.boxplot(x = 'day',y = 'total_bill',hue = 'time',data = tips)

# Draw a boxplot with nested grouping when some bins are empty:

sns.set_style("whitegrid")
sns.boxplot(x = 'day',y = 'total_bill',hue = 'time',data = tips,linewidth = 3.5)

# Draw a boxplot for each numeric variable in a DataFrame:
iris = sns.load_dataset("iris")
sns.boxplot(data = iris,orient ='h',palette = 'Set2')

# Use swarmplot() to show the datapoints on top of the boxes:
sns.boxplot(x = 'day',y = 'total_bill',data = tips)
sns.swarmplot(x = 'day',y = 'total_bill',data = tips,color = 'black')



