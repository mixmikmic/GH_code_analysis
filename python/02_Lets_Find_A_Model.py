from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

ml_models = {
    'LR': LogisticRegression(),
    'SGD': SGDClassifier(tol=1e-3, max_iter=1000),
    'LDA': LinearDiscriminantAnalysis(),
    'DT': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(),
    'NB': GaussianNB(),
    'SVM': SVC(),
    'NN': MLPClassifier()
}

import pandas

training_dataset = pandas.read_hdf('data/training_data.h5', key='trimer')
X = training_dataset[['orient0', 'orient1', 'orient2', 'orient3', 'orient4', 'orient5']].values
Y = training_dataset['class'].values
classes = training_dataset['class'].unique()

from sklearn import model_selection

validation_size = 0.20
seed = 7

selected = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
X_train, X_validation, Y_train, Y_validation = selected

scoring='accuracy'
n_splits = 2
# Typicall n_splits would be 10 but it runs much slower
#n_splits = 10

# Iterate through each model in our dictionary of models
for name, model in ml_models.items():
    kfold = model_selection.KFold(n_splits=n_splits, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    print(f'{name:5s}: {cv_results.mean():.5f} ± {cv_results.std():.5f}')

import itertools

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

get_ipython().run_line_magic('matplotlib', 'inline')

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

knn = ml_models['KNN']
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
plot_confusion_matrix(confusion_matrix(Y_validation, predictions, labels=classes), classes, normalize=True)

svm = ml_models['SVM']
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validation)
plot_confusion_matrix(confusion_matrix(Y_validation, predictions, labels=classes), classes, normalize=True)

nn = ml_models['NN']
nn.fit(X_train, Y_train)
predictions = nn.predict(X_validation)
plot_confusion_matrix(confusion_matrix(Y_validation, predictions, labels=classes), classes, normalize=True)

from sklearn.externals import joblib
joblib.dump(ml_models['KNN'], 'data/knn-model.pkl') 

