from sklearn import svm
from sklearn import datasets
clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)  

import pickle
pickle_out = open("pickle/test_classifier.pkl", "wb")
pickle.dump(clf, pickle_out)
pickle_out.close()

pickle_in = open("pickle/test_classifier.pkl", "rb")
# This will load the object from the pickle
pickled_classifier = pickle.load(pickle_in)
pickle_in.close()

pickled_classifier

from sklearn.externals import joblib
joblib.dump(clf, 'pickle/test_joblib_classifier.pkl') 

clf = joblib.load('pickle/test_joblib_classifier.pkl') 

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

pipe = Pipeline([('pca', PCA()),
                 ('svc', svm.SVC(C=10))])
pipe.fit(X, y)

joblib.dump(pipe, 'pickle/pipe.pkl') 

pipe_pickled = joblib.load('pickle/pipe.pkl') 

print(pipe_pickled.steps[0][1])
print(pipe_pickled.steps[1][1])

