import pandas as pd
df = pd.read_csv('../Lecture_6-AppliedMachineLearning/data/winequality-white.csv', sep=';')
df.info()

# Prepare X and y...
features = list(df.columns[:-1])
X = df[features]
y = df["quality"]

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

import graphviz
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data)  
graph

from sklearn.metrics import confusion_matrix

class_names = range(1, 11)
prediction = clf.predict(X)
cm = confusion_matrix(y, prediction, class_names)
cm

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
plt.matshow(cm)
plt.colorbar()

import itertools
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

# Plot out confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=class_names)

from sklearn.metrics import accuracy_score
accuracy_score(y, prediction)





