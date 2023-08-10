from mlsql import repl, execute

from sklearn import svm
from sklearn import datasets

#Train SVM
clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)

#Joblib
from sklearn.externals import joblib
joblib.dump(clf, 'filename.pkl')
clf = joblib.load('filename.pkl')

#Show current model
clf

from mlsql.functions.utils.modelIO import save_model, load_model
save_model("example.txt", clf)

with open("example.txt.mlsql", "r") as f:
    text = f.read()
    print(text)

new_model = load_model("example.txt.mlsql")

new_model

# Iris Dataset exmaple with language

command = "LOAD /home/ubuntu/notebooks/ML-SQL/dataflows/Classification/iris.data ()"

execute(command)

