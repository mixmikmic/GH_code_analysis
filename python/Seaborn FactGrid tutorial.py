import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

tips_data = sns.load_dataset("tips")
tips_data.head()



# Initialize a 2x2 grid of facets using the tips dataset:
sns.set(style="ticks", color_codes=True)
sns.FacetGrid(tips_data,row = 'time',col = 'smoker')


# Draw a univariate plot on each facet:
x = sns.FacetGrid(tips_data,col = 'time',row = 'smoker')
x = x.map(plt.hist,"total_bill")

bins = np.arange(0,65,5)
x = sns.FacetGrid(tips_data, col="time",  row="smoker")
x =x.map(plt.hist, "total_bill", bins=bins, color="g")

# Plot a bivariate function on each facet:

x = sns.FacetGrid(tips_data, col="time",  row="smoker")
x = x.map(plt.scatter, "total_bill", "tip", edgecolor="w")

# Assign one of the variables to the color of the plot elements:

x = sns.FacetGrid(tips_data, col="time",  hue="smoker")
x = x.map(plt.scatter,"total_bill","tip",edgecolor = "w")
x =x.add_legend()

# Change the size and aspect ratio of each facet:

x = sns.FacetGrid(tips_data, col="day", size=4, aspect=.5)
x =x.map(plt.hist, "total_bill", bins=bins)

# Specify the order for plot elements:

g = sns.FacetGrid(tips_data, col="smoker", col_order=["Yes", "No"])
g = g.map(plt.hist, "total_bill", bins=bins, color="m")

# Use a different color palette:

kws = dict(s=50, linewidth=.5, edgecolor="w")
g =sns.FacetGrid(tips_data, col="sex", hue="time", palette="Set1",                   hue_order=["Dinner", "Lunch"]) 

g = g.map(plt.scatter, "total_bill", "tip", **kws)
g.add_legend()

# Use a dictionary mapping hue levels to colors:

pal = dict(Lunch="seagreen", Dinner="gray")
g = sns.FacetGrid(tips_data, col="sex", hue="time", palette=pal,                   hue_order=["Dinner", "Lunch"])

g = g.map(plt.scatter, "total_bill", "tip", **kws)
g.add_legend()

# FacetGrid with boxplot
x = sns.FacetGrid(tips_data,col= 'day')
x = x.map(sns.boxplot,"total_bill","time")



