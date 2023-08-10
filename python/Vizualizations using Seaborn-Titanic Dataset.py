import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

sns.set_style('whitegrid')

titanic = sns.load_dataset('titanic')

titanic.head()

sns.jointplot(x='fare',y='age',data=titanic,kind='scatter')

#couting the distribution of Fare
sns.distplot(titanic['fare'],kde=False)

#playing with Swarm Plots to analyze class vs age
sns.swarmplot(x="class", y="age", data=titanic)

#count plot to analyze the distribution of passengers by sex
sns.countplot(x='sex',data=titanic)

#making correlation matrix for titanic dataset
t=titanic.corr()

t

#drawing a heatmap to check which parameters are most valuable and correlated for Analysis
sns.heatmap(t)

#Multiple plots using grid plot functions
g = sns.FacetGrid(titanic, col="sex")
g = g.map(plt.hist, "age")

