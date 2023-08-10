get_ipython().magic("config InlineBackend.figure_format='retina'")
get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams["font.size"] = 14

churn_df = pd.read_csv('data/churn.csv')

churn_df.head().T

churn_df.describe().T

y = np.where(churn_df['Churn?'] == 'True.', 1, 0)
print('Fraction of users leaving:', y.mean())

to_drop = ['State', 'Area Code', 'Phone', 'Churn?']
churn_features = churn_df.drop(to_drop, axis=1)

yes_no_columns = ["Int'l Plan", "VMail Plan"]
churn_features[yes_no_columns] = churn_features[yes_no_columns] == 'yes'

churn_features.head().T

# convert everything to numerical values
X = churn_features.as_matrix().astype(np.float)

from sklearn.cross_validation import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier


classifiers = [('Dummy', DummyClassifier(strategy='most_frequent')),
               ('RF', RandomForestClassifier()),
               ('kNN', KNeighborsClassifier())]

for name,classifier in classifiers:
    scores = cross_val_score(classifier, X, y, scoring='accuracy')
    print(name, 'scores:', scores)

from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import cross_val_predict
import matplotlib.pyplot as plt

classifiers = [('Dummy', DummyClassifier(strategy='most_frequent')),
               ('RF', RandomForestClassifier()),
               ('kNN', KNeighborsClassifier())]

def plot_confusion_matrix(cm, ax, title='Confusion matrix', cmap=plt.cm.summer_r):
    ax.imshow(cm, interpolation='none', cmap=cmap)

    ax.set_xticks([0,1])
    ax.set_yticks([0,1])

    ax.set_xticklabels(['stay', 'churn'])
    ax.set_yticklabels(['stay', 'churn'])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.annotate(str(cm[i][j]), xy=(j, i), color='black')

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title(title)


fig, axarr = plt.subplots(nrows=len(classifiers), figsize=(7, 7*len(classifiers)))

for (name,classifier), ax in zip(classifiers, axarr):
    y_pred = cross_val_predict(classifier, X, y)
    plot_confusion_matrix(confusion_matrix(y, y_pred), ax,
                          'Confusion matrix (%s)'%name)
    
plt.tight_layout()

from sklearn.cross_validation import StratifiedKFold
from utils import draw_roc_curve

clf = RandomForestClassifier()
#clf = RandomForestClassifier(n_estimators=30)
#clf = RandomForestClassifier(max_depth=5)
cv = StratifiedKFold(y, n_folds=3)
    
draw_roc_curve(clf, cv, X, y)

