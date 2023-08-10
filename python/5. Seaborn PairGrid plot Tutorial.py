import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

iris_data = sns.load_dataset("iris")
x = sns.PairGrid(iris_data)
x = x.map(plt.scatter)



# Show a univariate distribution on the diagonal:
x = sns.PairGrid(iris_data)
x = x.map_diag(plt.hist)
x = x.map_offdiag(plt.scatter)

# Color the points using a categorical variable:
x = sns.PairGrid(iris_data,hue = 'species')
x = x.map_diag(plt.hist)
x = x.map_offdiag(plt.scatter)
x = x.add_legend()



# Use a different style to show multiple histograms:
x = sns.PairGrid(iris_data,hue = 'species')
x = x.map_diag(plt.hist,histtype = 'step',linewidth =3)
x = x.map_offdiag(plt.scatter)
x = x.add_legend()

# Plot a subset of variables
x = sns.PairGrid(iris_data, vars=["sepal_length", "sepal_width"])
x =  x.map(plt.scatter)

x = sns.PairGrid(iris_data,hue = 'species',vars = ['petal_length','petal_width'])
x = x.map_diag(plt.hist)
x = x.map_offdiag(plt.scatter)
x = x.add_legend()

# Use different variables for the rows and columns:
x = sns.PairGrid(iris_data,x_vars=["sepal_length", "sepal_width"],                 y_vars=["petal_length", "petal_width"])
x = x.map(plt.scatter)

## Use different functions on the upper and lower triangles:

x = sns.PairGrid(iris_data)
x = x.map_diag(plt.hist)
x = x.map_upper(plt.scatter)
x = x.map_lower(sns.kdeplot)
x = x.add_legend()

# Use different colors and markers for each categorical level:

g = sns.PairGrid(iris_data, hue="species", palette="Set2",
                 hue_kws={"marker": ["o", "s", "D"]})
g = g.map(plt.scatter, linewidths=1, edgecolor="w", s=40)
g = g.add_legend()



