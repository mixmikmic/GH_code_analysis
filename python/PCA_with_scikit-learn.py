import urllib.request
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
file_name = "iris_data.txt"
urllib.request.urlretrieve(url, file_name)

iris_table = pd.read_csv(file_name, names=["sepal length", "sepal_width", "petal_length", "petal_width", "iris_class"])

iris_table.head()

X_iris = iris_table.loc[: , ["sepal length", "sepal_width", "petal_length", "petal_width"]].values

X_iris

pca = PCA(n_components=2)

principal_components = pca.fit_transform(X_iris)

principal_components

colors = iris_table.iris_class.apply(
    lambda iris_class: 
    {"Iris-setosa": 0, 
     "Iris-versicolor": 1, 
     "Iris-virginica": 2}[iris_class])

plt.scatter(principal_components[:, 0], principal_components[:, 1], c=colors)

