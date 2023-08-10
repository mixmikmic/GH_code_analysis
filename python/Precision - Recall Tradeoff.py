get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC


digits = load_digits()
X, y = digits.data, digits.target % 2

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

svm = LinearSVC(random_state=42).fit(X_train, y_train)
y_pred = svm.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=["even", "odd"]))

decision_function = svm.decision_function(X_test)
y_pred_2 = decision_function > -2
print(classification_report(y_test, y_pred_2, target_names=["even", "odd"]))

from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, decision_function)
plt.plot(precision, recall)
plt.xlabel("precision")
plt.ylabel("recall")

from sklearn.metrics import roc_curve

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, decision_function)
plt.plot(false_positive_rate, true_positive_rate)
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.svm import SVC

X, y = digits.data, digits.target == 3
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


for gamma in [.01, .05, 1]:
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (recall)")
    svm = SVC(gamma=gamma).fit(X_train, y_train)
    decision_function = svm.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, decision_function)
    acc = svm.score(X_test, y_test)
    auc = roc_auc_score(y_test, svm.decision_function(X_test))
    plt.plot(fpr, tpr, label="gamma: %.2f accuracy:%.2f auc:%.2f" % (gamma, acc, auc), linewidth=3)
plt.legend(loc=(1, 0))





