from sklearn import svm
from sklearn import datasets
from sklearn.externals import joblib
import data_wrangler

iris = datasets.load_iris()
_data, _label = iris.data, iris.target

_data = data_wrangler.pre_process_data(_data)

clf = svm.SVC()
clf.fit(_data, _label)

joblib.dump(clf, 'iris_clf_model.pkl')



