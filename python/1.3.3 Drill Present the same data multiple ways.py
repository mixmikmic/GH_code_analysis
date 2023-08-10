import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Load CSV

wine = pd.read_csv('winequality-red.csv')
for wine_type in wine:
    wine['type'] = 'Red'

wine.head()

#1 Histogram

plt.hist(wine['volatile acidity'], bins=20, color='red', alpha=.5)
plt.xlabel('Volatile Acidity')
plt.title('Plot 1: Histogram of Volatile Acidity')
plt.show()

#2 Boxplot

sns.set_style("whitegrid")
sns.boxplot(wine['volatile acidity'])
plt.title('Plot 2: Boxplot of Volatile Acidity')
sns.despine(offset=10)
plt.show()

#3 Barplot

sns.set_style("darkgrid")
sns.barplot(y="volatile acidity", data=wine, palette="Dark2_r")
plt.title('Plot 3: Barplot of Volatile Acidity')
sns.despine(offset=10)
plt.show()

#4 Pointplot

sns.set(style="whitegrid")
sns.factorplot(y='volatile acidity', data=wine, size=5, kind="point", palette="pastel", ci=95, dodge=True)
sns.despine(left=True)
plt.title('Plot 4: Pointplot of Volatile Acidity')
plt.show()

#Scatterplot
w = sns.lmplot(y='fixed acidity', x='alcohol', data=wine, fit_reg=False, scatter_kws={'alpha':0.4})
w.set_ylabels("Fixed Acidity")
w.set_xlabels("Alcohol")
plt.title('Plot 1: Scatterplot - Fixed acidity by alcohol')
plt.show()

#Scatterplot with regression line
w = sns.lmplot(y='fixed acidity', x='alcohol', data=wine, fit_reg=True, scatter_kws={'alpha':0.4})
w.set_ylabels("Fixed Acidity")
w.set_xlabels("Alcohol")
plt.title('Plot 2: Scatterplot with regression line - Fixed acidity by alcohol')
plt.show()

#Joint kernel density estimate
w = sns.jointplot(y='fixed acidity', x='alcohol', data=wine, kind="kde", size=7, space=0)
plt.title('                                                                     Plot 3: Joint kernel density estimate - Fixed acidity by alcohol')
plt.show()

# Histogram
plt.hist(wine['total sulfur dioxide'], bins=20, color='red',  alpha=.5, label='Red')
plt.xlabel('Total sulfur dioxide')
plt.title('Plot 1: Histogram - Total sulfur dioxide')
plt.show()

# Swarmplot scatterpltot
sns.set(style='whitegrid', color_codes=True)
sns.swarmplot(x='type', y='total sulfur dioxide', data=wine);
plt.title('Plot 2: Swarmplot - Total sulfur dioxide')
plt.show()

# Barplot
g=sns.factorplot(x='type', y='total sulfur dioxide', hue='type', data=wine, kind='bar', palette='CMRmap', ci=95)
plt.title('Plot 3: Barplot - Total sulfur dioxide for red wines')
plt.show()

# Boxplot
sns.boxplot(x='type', y='total sulfur dioxide', hue='type', data=wine)
plt.title('Plot 4: Boxplot - Total sulfur dioxide for red wines')
sns.despine(offset=5, trim=True)
plt.show()

# Violin pointplot
sns.set(style="whitegrid")
g = sns.factorplot(x='type', y='total sulfur dioxide', hue='type', data=wine, kind='violin', ci=95, dodge=True, join=False)
plt.title('Plot 5: Violin pointplot - Total sulfur dioxide for red wine')
plt.show()

# Stripplot scatterpltots
sns.set(style='whitegrid', color_codes=True)
sns.stripplot(x='type', y='total sulfur dioxide', data=wine);
plt.title('Plot 6: Strip plot - Total sulfur dioxide')
plt.show()

