import matplotlib as mpl
get_ipython().magic('matplotlib inline')
mpl.rcParams['figure.figsize'] = (10.0, 10.0)

# Print rcParams

from matplotlib import pyplot as plt
import pandas as pd

df = pd.DataFrame.from_csv('datasets/iris_data.csv', index_col=4)

# Plot using pandas dataframe function

df.plot()

# Plot using pyplot

plt.plot(df)

# Creates a figure and plots a scatter plot on x and y inputs

df.plot(kind='scatter', x='sepal_length', y='sepal_width', color='DarkBlue')
# Provides title for plot
plt.title('Sepal Length vs. Sepal Width') 

# Swtich to interactive plotting
get_ipython().magic('matplotlib')

# Set up figure object and plot first data group
ax = df.ix['Iris-setosa'].plot(kind='scatter', x='sepal_length', y='sepal_width', color='DarkBlue', label='Iris-setosa')

# Add second group subplot to ax object and plot
df.ix['Iris-virginica'].plot(kind='scatter', x='sepal_length', y='sepal_width', color='Cyan', label='Iris-virginica', ax=ax)

# Add third group subplot to ax object and plot
df.ix['Iris-versicolor'].plot(kind='scatter', x='sepal_length', y='sepal_width', color='White', label='Iris-versicolor', ax=ax)

# Provides title for plot
plt.title('Sepal Length vs. Sepal Width')

plt.figure('hist #1')
plt.title('Petal Length with 50 Bins')
plt.hist(df['petal_length'], bins=50)

plt.figure('KDE')
df['petal_length'].plot(kind='kde')

from pandas.tools.plotting import scatter_matrix

scatter_matrix(df)

df.plot(kind='box')

import seaborn as sns
sns.set(style="white")
plt.figure('Heatmap')
groups = df.groupby(df.index)
sns.heatmap(groups.mean(), annot=True, linewidths=.5)
plt.title('Heatmap by Average Values')

plt.figure('Petal Length with KDE and Histogram')
plt.title('Distribution of Petal Length Histogram and KDE')
sns.distplot(df['petal_length'], kde=True, color="b", bins=50)
sns.jointplot(df['sepal_length'],df['sepal_width'], kind='kde', size=7, space=0)
sns.jointplot(df['petal_length'],df['petal_width'], kind='kde', size=7, space=0)

# Need to add species to a column as seaborn doesn't take index
df['species']=df.index

# Seaborn calls the scatter matrix a pairplot.
sns.set(style='darkgrid')
sns.pairplot(df, hue='species')

import numpy as np

# Use numpy to build a vector of 100 equally spaced points between -20 and 20
x = np.linspace(-20,20,500)

# Use numpy to calculate the sin of the x vector
y = np.sin(x)

# Plot the individual points using blue ('b') circles ('o')
plt.figure('Functions')
plt.plot(x,y,'bo')

# Plot the line through the circles
plt.plot(x,y)

plt.figure('Plots of Sepals and Petals')
plt.subplots_adjust(hspace=.4)
plt.subplot(2,1,1)
plt.title('Sepal Measurements')
plt.scatter(x=df['sepal_length'], y=df['sepal_width'])
plt.grid()
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.subplot(2,1,2)
plt.title('Petal Measurements')
plt.scatter(x=df['petal_length'], y=df['petal_width'])
plt.grid()
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

plt.figure('Plots of Sepals and Petals with Shared X Axis')
plt.subplots_adjust(hspace=.4)
ax = plt.subplot(2,1,1)
plt.title('Sepal Measurements')
plt.scatter(x=df['sepal_length'], y=df['sepal_width'])
plt.grid()
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.subplot(2,1,2, sharex=ax)
plt.title('Petal Measurements')
plt.scatter(x=df['petal_length'], y=df['petal_width'])
plt.grid()
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

