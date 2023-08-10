from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import pickle
#conda install requests will be needed
import requests, json


# import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target

print(iris.DESCR)

X_train, X_test, y_train, y_test = train_test_split(X, y)

rfc = RandomForestClassifier(n_estimators=100, n_jobs=2)

rfc.fit(X_train, y_train)

print("Accuracy = %0.2f" % accuracy_score (y_test, rfc.predict(X_test)))
print(classification_report(y_test, rfc.predict(X_test)))

pickle.dump(rfc, open("iris_rfc.pkl", "wb"))

my_random_forest = pickle.load(open("iris_rfc.pkl", "rb"))

my_random_forest

print(classification_report(y_test, my_random_forest.predict(X_test)))


url = "http://localhost:9000/api"
data = json.dumps({'sl':5.84,'sw':3.0, 'pl':3.75,'pw':1.1}) 
r = requests.post(url, data)

print(r.json())



