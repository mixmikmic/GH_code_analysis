import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

iris = sns.load_dataset('iris')

iris.head()

# Just the Grid
sns.PairGrid(iris)

# Then you map to the grid
g = sns.PairGrid(iris)
g.map(plt.scatter)

# Map to upper,lower, and diagonal
g = sns.PairGrid(iris)
g.map_diag(plt.hist)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)

sns.pairplot(iris)

sns.pairplot(iris,hue='species',palette='rainbow')

tips = sns.load_dataset('tips')

tips.head()

# Just the Grid
g = sns.FacetGrid(tips, col="time", row="smoker")

g = sns.FacetGrid(tips, col="time",  row="smoker")
g = g.map(plt.hist, "total_bill")

g = sns.FacetGrid(tips, col="time",  row="smoker",hue='sex')
# Notice hwo the arguments come after plt.scatter call
g = g.map(plt.scatter, "total_bill", "tip").add_legend()

g = sns.JointGrid(x="total_bill", y="tip", data=tips)

g = sns.JointGrid(x="total_bill", y="tip", data=tips)
g = g.plot(sns.regplot, sns.distplot)

