import scipy
import numpy as np
from scipy import interp
from sklearn import preprocessing, cross_validation, neighbors,datasets, svm
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

fileName = 'HOGLBP'
x = scipy.io.loadmat(fileName)
cc = [x][0]
HOGLBP = cc['HOGLBP']
HOGLBP = np.asarray(HOGLBP)
print "HOGLBP dataset loaded"

fileName = 'LBPlabels'
x = scipy.io.loadmat(fileName)
cc = [x][0]
LBPlabels = cc['WW']
LBPlabels = np.asarray(LBPlabels)
print 'labels loading completed'

c, r = LBPlabels.shape
LBPlabels = LBPlabels.reshape(c,)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(HOGLBP, LBPlabels, test_size=0.3, random_state=0)

#scaling
scaler = StandardScaler()
#scaler = preprocessing.MinMaxScaler()

# Fit only on training data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
# apply same transformation to test data
X_test = scaler.transform(X_test)

clf = AdaBoostClassifier(n_estimators=500)
scores = cross_val_score(clf, X_train, y_train)
scores.mean()
clf.fit(X_train, y_train)

Accuracy = clf.score(X_train, y_train)
print "Accuracy in the training data: ", Accuracy*100, "%"

accuracy = clf.score(X_test, y_test)
print "Accuracy in the test data", accuracy*100, "%"

y_pred = clf.predict(X_train)
print '\nTraining classification report\n', classification_report(y_train, y_pred)
print "\n Confusion matrix of training \n", confusion_matrix(y_train, y_pred)

y_pred = clf.predict(X_test)
print '\nTesting classification report\n', classification_report(y_test, y_pred)
print "\nConfusion matrix of the testing\n", confusion_matrix(y_test, y_pred)

probas = clf.fit(X_train, y_train).predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)

mean_tpr += interp(mean_fpr, fpr, tpr)
mean_tpr[0] = 0.0
roc_auc = auc(fpr, tpr)
print "\nArea Under the ROC curve: ", roc_auc

meanTP = 0
for t in tpr:
    meanTP += t
print "Mean True Positive rate (testing): ", meanTP/len(tpr)

meanFP = 0
for t in fpr:
    meanFP += t
print "Mean False Positive rate (testing): ", meanFP/len(fpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

from sklearn.externals import joblib
joblib.dump(clf, 'AdaBoostTBModel.pkl')
print "AdaBoostTBModel Generated"

#X is the sample you want to classify (HOG &LBP features of an x-ray)
#Image size used in the training is 100*100pixels
#You can pass one image or a numpy array of more than one
#to use the model for testing uncomment the following lines:

#clf = joblib.load('AdaBoostTBModel.pkl') 
#clf.predict(X)



