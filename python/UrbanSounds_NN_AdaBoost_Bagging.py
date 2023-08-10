import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skflow

from collections import Counter
from sklearn import datasets, metrics, cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

get_ipython().magic('matplotlib inline')

def plotHeatmap(y_actual, y_pred, y_labels, printCF=False):
    y_counts = np.asarray(Counter(y_actual).values())
    confMat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    if printCF:
        print(confMat)
    confMatNorm = np.true_divide(confMat, y_counts)
    
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(confMatNorm, cmap=plt.cm.gray, vmin=0, vmax=1)

    # locate ticks
    ax.set_xticks(np.arange(confMatNorm.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(confMatNorm.shape[1])+0.5, minor=False)

    # move x ticks to top
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(y_labels, minor=False, rotation='vertical')
    ax.set_yticklabels(y_labels, minor=False)

    #ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Actual Class')

    plt.show()

# Import Pre-Processed Wav File Data Set
wavData = pd.read_csv('feature_quant_new.csv')

# Remove Empty Rows
wavData = wavData[-np.isnan(wavData['mean'])]

print len(wavData)
print wavData[0:2]

# create feature (only) array by dropping index and class columns
feat = list(wavData.columns)
feat.remove('class')
feat.remove('Unnamed: 0')

# convert class labels to numeric category 0-9
le = LabelEncoder()
y = wavData.loc[:,'class']
y = le.fit_transform(y)
y_labels = list(le.classes_)

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(wavData.loc[:,feat], y,                                                     test_size=0.3, random_state=0)
X_train.shape

# scale data according to training set distribution
sc = StandardScaler()
sc=sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

classifier = skflow.TensorFlowDNNClassifier(hidden_units=[7], n_classes=10, steps=10000,                                             tf_random_seed=0, learning_rate = 0.1)
classifier.fit(X_train_std, y_train)
score = metrics.accuracy_score(y_test, classifier.predict(X_test_std))
print("Accuracy: %f" % score)

print(f1_score(y_test, classifier.predict(X_test_std), average='micro'))
print(f1_score(y_test, classifier.predict(X_test_std), average='macro'))
print(f1_score(y_test, classifier.predict(X_test_std), average='weighted'))

plotHeatmap(y_test, classifier.predict(X_test_std), y_labels, printCF=True)

tree = DecisionTreeClassifier(criterion='entropy',
                             max_depth=6)
ada = AdaBoostClassifier(base_estimator=tree,
                        n_estimators=2000,
                        learning_rate=0.1,
                        random_state=0)
ada = ada.fit(X_train_std, y_train)
y_train_pred = ada.predict(X_train_std)
y_test_pred = ada.predict(X_test_std)
ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)
print('AdaBoost train/test accuracies %.3f/%.3f'
     % (ada_train, ada_test))

print(f1_score(y_test, y_test_pred, average='micro'))
print(f1_score(y_test, y_test_pred, average='macro'))
print(f1_score(y_test, y_test_pred, average='weighted'))

plotHeatmap(y_test, y_test_pred, y_labels, printCF=True)

tree = DecisionTreeClassifier(criterion='entropy', max_depth=None)
bag = BaggingClassifier(base_estimator=tree,n_estimators=500,max_samples=0.5,max_features=0.5,
                       bootstrap=True,bootstrap_features=False,n_jobs=1, random_state=1)

bag = bag.fit(X_train_std, y_train)
y_train_pred = bag.predict(X_train_std)
y_test_pred = bag.predict(X_test_std)
bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test, y_test_pred)
print('Bagging train/test accuracies %.3f/%.3f'
     % (bag_train, bag_test))

print(f1_score(y_test, y_test_pred, average='micro'))
print(f1_score(y_test, y_test_pred, average='macro'))
print(f1_score(y_test, y_test_pred, average='weighted'))

plotHeatmap(y_test, y_test_pred, y_labels, printCF=True)



