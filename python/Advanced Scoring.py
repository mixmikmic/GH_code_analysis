get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
np.set_printoptions(precision=2)

digits = load_digits()
X, y = digits.data, digits.target == 3
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.svm import SVC

from sklearn.cross_validation import cross_val_score
cross_val_score(SVC(), X_train, y_train)

from sklearn.dummy import DummyClassifier
cross_val_score(DummyClassifier("most_frequent"), X_train, y_train)



from sklearn.metrics import roc_curve, roc_auc_score

for gamma in [.01, .1, 1]:
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    svm = SVC(gamma=gamma).fit(X_train, y_train)
    decision_function = svm.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, decision_function)
    acc = svm.score(X_test, y_test)
    auc = roc_auc_score(y_test, svm.decision_function(X_test))
    plt.plot(fpr, tpr, label="acc:%.2f auc:%.2f" % (acc, auc))
    print()
plt.legend(loc="best")

from sklearn.metrics.scorer import SCORERS

SCORERS.keys()

def my_accuracy(est, X, y):
    return np.mean(est.predict(X) == y)

from sklearn.svm import LinearSVC
print(cross_val_score(LinearSVC(random_state=0), X, y, cv=5))
print(cross_val_score(LinearSVC(random_state=0), X, y, cv=5, scoring=my_accuracy))



