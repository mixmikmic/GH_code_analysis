import sknn.mlp
import sklearn.metrics
import pandas as pd
print 'Hi!'
import matplotlib.pyplot as plt

df= pd.read_csv('DataSets/New_Mod_1.csv',sep=',')

from sklearn.cross_validation import train_test_split

train, test = train_test_split(df, test_size = 0.3)

Y_train = train.pop('apnea')
X_train = train

Y_test= test.pop('apnea')
X_test= test

print X_train

from sklearn import preprocessing
X_train=X_train.values
X_test=X_test.values
Y_train=Y_train.values
Y_test=Y_test.values

X_train = preprocessing.MinMaxScaler().fit_transform(X_train)
X_test = preprocessing.MinMaxScaler().fit_transform(X_test)
Y_train = preprocessing.MinMaxScaler().fit_transform(Y_train)
Y_test = preprocessing.MinMaxScaler().fit_transform(Y_test)

for r in X_test:
    print r

from sknn.mlp import Classifier, Layer

nn = Classifier(
    layers=[
        Layer("Sigmoid", units=20),
        Layer("Sigmoid", units=10),
         Layer("Sigmoid", units=5),
        Layer("Softmax")],
    learning_rate=0.001,
    n_iter=100)
nn.fit(X_train, Y_train)

Y_test

score = nn.score(X_test, Y_test)

score

y_predicted = nn.predict(X_test)

print y_predicted

y_predic=y_predicted.flatten()

import numpy as np
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_predic,Y_test, pos_label=0)

fpr


tpr

import matplotlib.pyplot as plt
import numpy as np
# false_positive_rate 
# true_positive_rate 

# This is the ROC curve
plt.plot(tpr,fpr)
plt.show() 


# This is the AUC
roc_auc = np.trapz(fpr,tpr)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()







