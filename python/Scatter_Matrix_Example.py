get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris_data = pd.read_csv('https://raw.githubusercontent.com/rhiever/Data-Analysis-and-Machine-Learning-Projects/master/example-data-science-notebook/iris-data-clean.csv', na_values=['NA'])

iris_data.head()

iris_data.rename(columns={'class':'category'}, inplace=True)

sns.pairplot(iris_data.dropna(), hue='category')

iris_data.corr()

iris_data.cov()

sns.heatmap(iris_data.corr(), annot=True, fmt='0.2f')
plt.xticks(rotation=270)

from pandas.tools.plotting import scatter_matrix
plt.style.use('ggplot')

colors_palette = {'Iris-setosa':'red', 'Iris-versicolor':'green', 'Iris-virginica':'blue'}
colors = [colors_palette[c] for c in iris_data['category']]
scatter_matrix(iris_data.dropna(), figsize=(12,12), color=colors, diagonal='hist', grid=True)
plt.suptitle("Scatter Matrix Using Pandas", fontsize=30)
plt.show()

