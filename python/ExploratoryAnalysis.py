# Pandas for managing datasets
import pandas as pd
# Matplotlib for additional customization
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Seaborn for plotting and styling
import seaborn as sns

# Read dataset
df = pd.read_csv('data/winequality-red.csv', sep=';')

# Display first 5 observations
df.head()

sns.lmplot(data=df, y='quality', x='pH')

sns.lmplot(data=df, y='quality', x='chlorides')
sns.lmplot(data=df, y='quality', x='fixed acidity')

sns.lmplot(data=df, y='pH', x='chlorides', hue='quality', fit_reg=False)

plt.figure(figsize=(18,5))
sns.boxplot(data=df)

stats_df = df.drop(['total sulfur dioxide', 'free sulfur dioxide'], axis=1)

plt.figure(figsize=(18,5))
sns.boxplot(data=stats_df)

sulfur_df = df[['total sulfur dioxide', 'free sulfur dioxide']]
sulfur_df.head()

sns.boxplot(data=sulfur_df)

# Calculate correlations
corr = df.corr()
 
# Heatmap
sns.heatmap(corr)

sns.lmplot(data=df, y='quality', x='volatile acidity')

sns.lmplot(data=df, y='quality', x='alcohol')



