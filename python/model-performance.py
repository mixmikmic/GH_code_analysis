import itertools
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV

plt.rc('font', size=14)        # controls default text sizes
plt.rc('axes', titlesize=16)   # fontsize of the axes title
plt.rc('axes', labelsize=16)   # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)  # fontsize of the tick labels
plt.rc('ytick', labelsize=14)  # fontsize of the tick labels
plt.rc('legend', fontsize=14)  # legend fontsize

f = open('databall.pkl')
X, X_train, X_test, y, y_train, y_test = pickle.load(f)
f.close()

# This function prints and plots the confusion matrix.
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.grid(visible=False)

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '%d\n%.2f%%' % (cm[i, j], cm_norm[i, j]*100),
                 horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

models = OrderedDict()
attributes = ['TEAM_SRS','TEAM_SRS_AWAY']

model = LogisticRegression()
model.fit(X_train[attributes], y_train)
models['Logistic Regression'] = model
pred = model.predict(X_test[attributes])
plot_confusion_matrix(confusion_matrix(y_test, pred), ['Loss', 'Win'])
print '_' * 52
print classification_report(y_test, pred, target_names=['Loss', 'Win'], digits=3)
print 'Correctly predicted %.2f%% of games' % (accuracy_score(y_test, pred) * 100)

model = LinearSVC()
model.fit(X_train[attributes], y_train)
pred = model.predict(X_test[attributes])

model = CalibratedClassifierCV(model, cv='prefit')
model.fit(X_train[attributes], y_train)
models['Support Vector Machine'] = model

plot_confusion_matrix(confusion_matrix(y_test, pred), ['Loss', 'Win'])
print '_' * 52
print classification_report(y_test, pred, target_names=['Loss', 'Win'], digits=3)
print 'Correctly predicted %.2f%% of games' % (accuracy_score(y_test, pred) * 100)

model = RandomForestClassifier(n_estimators=10)
model.fit(X_train[attributes], y_train)
models['Random Forest'] = model
pred = model.predict(X_test[attributes])
plot_confusion_matrix(confusion_matrix(y_test, pred), ['Loss', 'Win'])
print '_' * 52
print classification_report(y_test, pred, target_names=['Loss', 'Win'], digits=3)
print 'Correctly predicted %.2f%% of games' % (accuracy_score(y_test, pred) * 100)

model = MLPClassifier(hidden_layer_sizes=5)
model.fit(X_train[attributes], y_train)
models['Neural Network'] = model
pred = model.predict(X_test[attributes])
plot_confusion_matrix(confusion_matrix(y_test, pred), ['Loss', 'Win'])
print '_' * 52
print classification_report(y_test, pred, target_names=['Loss', 'Win'], digits=3)
print 'Correctly predicted %.2f%% of games' % (accuracy_score(y_test, pred) * 100)

model = GaussianNB()
model.fit(X_train[attributes], y_train)
models['Naive Bayes'] = model
pred = model.predict(X_test[attributes])
plot_confusion_matrix(confusion_matrix(y_test, pred), ['Loss', 'Win'])
print '_' * 52
print classification_report(y_test, pred, target_names=['Loss', 'Win'], digits=3)
print 'Correctly predicted %.2f%% of games' % (accuracy_score(y_test, pred) * 100)

def plot_performance_curves(models, X, y):
    plt.figure(figsize=(16, 6))
    
    for label, model in models.items():
        prob = model.predict_proba(X)
        
        # Plot ROC curve
        ax = plt.subplot(121)
        fpr, tpr, thresholds = roc_curve(y, prob[:, 1])
        roc_auc = roc_auc_score(y, prob[:, 1])
        ax.plot(fpr, tpr, label='%s (Area = %0.2f)' % (label, roc_auc))
        
        # Plot precision/recall curve
        ax = plt.subplot(122)
        precision, recall, thresholds = precision_recall_curve(y, prob[:, 1])
        pr_auc = average_precision_score(y, prob[:, 1])
        ax.plot(recall, precision, label='%s (Area = %0.2f)' % (label, pr_auc))
        
    # Label axes and add legend
    ax = plt.subplot(121)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    
    ax = plt.subplot(122)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend()
    ax.set_ylim(0.4)
    
    plt.show()

plot_performance_curves(models, X_test[attributes], y_test)

