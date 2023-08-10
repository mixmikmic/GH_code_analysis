import numpy as np
import pandas as pd

data = pd.read_csv("./formatted_data.csv",header=0, index_col=False)
data.head()

drop_cols = ['Sensor_'+x+'1' for x in map(chr,range(65,81))]
drop_cols.append('Batch_No')
data = data.drop(drop_cols, axis=1)
data.head()

data.describe()

from sklearn import preprocessing
target = data['Label']
data = data.drop('Label', axis=1)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
data_scaled = min_max_scaler.fit_transform(data)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_scaled, target, test_size=0.25, random_state=0)

from sklearn import tree
dt_classifier = tree.DecisionTreeClassifier()
dt_classifier = dt_classifier.fit(X_train, y_train)
y_pred = dt_classifier.predict(X_test)
print "Accuracy: %0.2f" %dt_classifier.score(X_test, y_test)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
get_ipython().magic('matplotlib inline')

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
   
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return


#plt.figure()
plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes=["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"],
                      title='Confusion matrix')

from sklearn.metrics import classification_report
print classification_report(y_test, y_pred, target_names=["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"])

from sklearn.model_selection import cross_val_score

def cv_score(clf,k):
    f1_scores = cross_val_score(clf, data_scaled, target, cv=k, scoring='f1_macro')
    print f1_scores
    print("F1 score: %0.2f (+/- %0.2f)" % (f1_scores.mean(), f1_scores.std() * 2))
    return

cv_score(dt_classifier,10)

from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=5)
#rf_classifier = rf_classifier.fit(X_train, y_train)
#y_pred_rf = classifier.predict(X_test)
cv_score(rf_classifier,10)

#plot_confusion_matrix(confusion_matrix(y_test, y_pred_rf), classes=["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"],
#                      title='Confusion matrix')

#print classification_report(y_test, y_pred_rf, target_names=["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"])

from sklearn import svm

svm_classifier = svm.SVC(C=1.0, kernel='rbf', gamma='auto', cache_size=9000, decision_function_shape = 'ovr')
cv_score(svm_classifier,10)

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)

grid.fit(data, target)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

