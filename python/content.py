get_ipython().magic('matplotlib inline')
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
from MNISTData import MNISTData
from sklearn.metrics import confusion_matrix

mnist = MNISTData(train_dir='MNIST_data', one_hot=True)

plt.gray()
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(mnist.train['images'][i].reshape((28,28)))
    plt.title(mnist.train['labels'][i].argmax())
    plt.axis('off')
plt.show()

def evaluate_classifier(clf, test_data, test_labels):
    pred = clf.predict(test_data)
    C = confusion_matrix(pred.argmax(axis=1), test_labels.argmax(axis=1))
    return C.diagonal().sum()*100./C.sum(),C

from sklearn import tree

clf = tree.DecisionTreeClassifier()
# --- your code here --- #

s,C = evaluate_classifier(clf, mnist.test['images'], mnist.test['labels'])
print s
print C

import numpy as np
from sklearn.cross_validation import StratifiedKFold

# --- your code here --- #

from sklearn import ensemble

clf = ensemble.RandomForestClassifier()
# --- your code here --- #

