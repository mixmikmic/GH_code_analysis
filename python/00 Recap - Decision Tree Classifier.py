from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

iris = datasets.load_iris()
digits = datasets.load_digits()

print(iris.DESCR[:172] + ' ...')
print(iris.feature_names)
print(iris.data[45:54])
print(iris.target[45:54])
print(iris.target_names)

lfeat = iris.feature_names
df_iris = pd.DataFrame(iris.data, columns = lfeat)
model = DecisionTreeClassifier()
data = df_iris[lfeat].values
df_iris["Species"] = iris.target
target = df_iris["Species"].values
model.fit(data, target)
expected = target
predicted = model.predict(data)
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))



