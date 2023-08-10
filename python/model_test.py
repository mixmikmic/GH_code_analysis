import joblib as jl
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dataset = jl.load('dataset/dataset_com.changba_xy.jl')

k = 5000

x_train = dataset['x'][:k]
y_train = dataset['y'][:k]

x_test = dataset['x'][k:]
y_test = dataset['y'][k:]

model = DecisionTreeClassifier()
model.fit(x_train, y_train)
model.score(x_test, y_test)







