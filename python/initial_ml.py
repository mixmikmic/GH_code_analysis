## Importing required libraries
import pandas as pd ## For DataFrame operation
import numpy as np ## Numerical python for matrix operations
from sklearn.model_selection import KFold, train_test_split ## Creating cross validation sets
from sklearn import metrics ## For loss functions
import matplotlib.pyplot as plt

## Libraries for Classification algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
import xgboost as xgb 
import lightgbm as lgb 

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import itertools
get_ipython().run_line_magic('matplotlib', 'inline')

train = pd.read_csv('balanced_train.csv')

train.shape

X = train.copy()

train.loc[ train.is_churn == 'no_churn', 'is_churn'] = 0
train.loc[ train.is_churn == 'churn', 'is_churn'] = 1

train['is_churn'] = train['is_churn'].astype(int)

X = X.drop('is_churn', axis = 1)
X = X.drop('amt_per_day', axis = 1)

y = train['is_churn']

def holdout_cv(X,y,size = 0.3, seed = 1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = seed)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = holdout_cv(X, y, size = 0.3, seed = 1)

X_test.head(1)

y_test.head(1)

X_test_just_1= X_test.head(1)
y_test_just_1= y_test.head(1)

X_test_just_1

y_test_just_1

check1 = model_dt.predict(X_test_just_1)

check1

### 2) Cross-Validation (K-Fold)
def kfold_cv(X,n_folds = 5, seed = 1):
    cv = KFold(n_splits = n_folds, random_state = seed, shuffle = True)
    return cv.split(X)

### Running Xgboost
def runXGB(train_X, train_y, test_X, test_y, seed_val=0, rounds=500, dep=8, eta=0.05,sub_sample=0.7,col_sample=0.7,
           min_child_weight_val=1, silent_val = 1):
    params = {}
    params["objective"] = "binary:logistic"
    params['eval_metric'] = 'auc'
    params["eta"] = eta
    params["subsample"] = sub_sample
    params["min_child_weight"] = min_child_weight_val
    params["colsample_bytree"] = col_sample
    params["max_depth"] = dep
    params["silent"] = silent_val
    params["seed"] = seed_val
    #params["max_delta_step"] = 2
    #params["gamma"] = 0.5
    num_rounds = rounds

    plst = list(params.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)


    xgtest = xgb.DMatrix(test_X, label=test_y)
    watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, 
                          early_stopping_rounds=100, verbose_eval=20)
    
    pred_test_y = model.predict(xgtest, ntree_limit=model.best_ntree_limit)
    
    loss = 0
    loss = metrics.roc_auc_score(test_y, pred_test_y)
    return pred_test_y, loss, model
 

pred_test_y_xg , loss_xg, model_xg = runXGB(X_train,y_train, X_test , y_test)

#pred_test_y_xg
accuracy_score(y_test, pred_test_y_xg)

clas = np.array(['churn','not_churn'])
clas

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("extra tree Normalized confusion matrix")
    else:
        print('extra tree Confusion matrix, without normalization')

    print(cm)

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

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, pred_test_y_xg)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=clas,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=clas, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

for i in range(len(pred_test_y_xg)):
    if pred_test_y_xg[i] >= 0.5:
       pred_test_y_xg[i] = 1
    else:
        pred_test_y_xg[i] = 0

plt.rcParams['figure.figsize'] = (7.0, 7.0)
xgb.plot_importance(booster=model_xg); plt.show()

X_imp = X.copy()

X_imp = X_imp.drop(['is_discount', 'is_cancel'], axis = 1)

X_train, X_test, y_train, y_test = holdout_cv(X_imp, y, size = 0.3, seed = 1)

pred_test_y_xg , loss_xg, model_xg = runXGB(X_train,y_train, X_test , y_test)

accuracy_score(y_test, pred_test_y_xg)

y_test.unique()

plt.rcParams['figure.figsize'] = (7.0, 7.0)
xgb.plot_importance(booster=model_xg); plt.show()

X_imp = X_imp.drop(['discount', 'notAutorenew_._cancel'], axis = 1)

X_train, X_test, y_train, y_test = holdout_cv(X_imp, y, size = 0.3, seed = 1)

pred_test_y_xg , loss_xg, model_xg = runXGB(X_train,y_train, X_test , y_test)

accuracy_score(y_test, pred_test_y_xg)

plt.rcParams['figure.figsize'] = (7.0, 7.0)
xgb.plot_importance(booster=model_xg); plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("extra tree Normalized confusion matrix")
    else:
        print('extra tree Confusion matrix, without normalization')

    print(cm)

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

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, pred_test_y_xg)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=clas,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=clas, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

xg_roc = roc_auc_score(y_test, pred_test_y_xg)
xg_fpr, xg_tpr, thresholds = roc_curve(y_test, pred_test_y_xg)

plt.figure()
plt.plot(xg_fpr, xg_tpr, label='xgboost (area = %0.2f)' % xg_roc)

# Plot Base Rate ROC
plt.plot([0,1], [0,1],label='Base Rate' 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()

def feature_importance(model,X):
    feature_importance = model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, X.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()

### Running Random Forest
def runRF(train_X, train_y, test_X, test_y, depth=20, leaf=10, feat=0.2):
    model = RandomForestClassifier(
            n_estimators = 1000,
                    max_depth = depth,
                    min_samples_split = 2,
                    min_samples_leaf = leaf,
                    max_features =  feat,
                    n_jobs = 4,
                    random_state = 0)
    model.fit(train_X, train_y)
    train_preds = model.predict_proba(train_X)[:,1]
    test_preds = model.predict_proba(test_X)[:,1]
    
    test_loss = 0
    
    train_loss = metrics.roc_auc_score(train_y, train_preds)
    test_loss = metrics.roc_auc_score(test_y, test_preds)
    print("Train and Test loss : ", train_loss, test_loss)
    return test_preds, test_loss, model

test_pred_rf, loss_rf, model_rf = runRF(X_train, y_train, X_test, y_test)

test_pred_rf

for i in range(len(test_pred_rf)):
    if test_pred_rf[i] >= 0.5:
       test_pred_rf[i] = 1
    else:
       test_pred_rf[i] = 0

accuracy_score(y_test, test_pred_rf)

feature_importance(model_rf,X_train)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("extra tree Normalized confusion matrix")
    else:
        print('extra tree Confusion matrix, without normalization')

    print(cm)

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

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, test_pred_rf)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=clas,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=clas, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

from sklearn.metrics import roc_auc_score
rf_roc = roc_auc_score(y_test,test_pred_rf )

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, model_rf.predict_proba(X_test)[:,1])

plt.figure()
plt.plot(fpr, tpr, label='random_forest (area = %0.2f)' % rf_roc)

# Plot Base Rate ROC
plt.plot([0,1], [0,1],label='Base Rate' 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()

plt.figure()
plt.plot(fpr, tpr, label='random_forest (area = %0.2f)' % rf_roc)
plt.plot(xg_fpr, xg_tpr, label='xgboost (area = %0.2f)' % xg_roc)
# Plot Base Rate ROC
plt.plot([0,1], [0,1],label='Base Rate' 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()

### Running Decision Tree
def runDT(train_X, train_y, test_X, test_y, criterion='gini', depth=None, min_split=2, min_leaf=1):
    model = DecisionTreeClassifier(criterion = criterion, max_depth = depth, 
                                   min_samples_split = min_split, min_samples_leaf=min_leaf)
    model.fit(train_X, train_y)
    train_preds = model.predict_proba(train_X)[:,1]
    test_preds = model.predict_proba(test_X)[:,1]
    
    test_loss = 0
    
    train_loss = metrics.roc_auc_score(train_y, train_preds)
    test_loss = metrics.roc_auc_score(test_y, test_preds)
    print("Train and Test loss : ", train_loss, test_loss)
    return test_preds, test_loss, model

test_pred_dt, loss_dt, model_dt = runDT( X_train, y_train, X_test, y_test)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("decision tree Normalized confusion matrix")
    else:
        print('decision tree Confusion matrix, without normalization')

    print(cm)

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

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, test_pred_rf)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=clas,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=clas, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

test_pred_dt

dt_roc = roc_auc_score(y_test,model_dt.predict_proba(X_test)[:,1])

dt_fpr, dt_tpr, thresholds = roc_curve(y_test, model_dt.predict_proba(X_test)[:,1])

plt.figure()
plt.plot(fpr, tpr, label='random_forest (area = %0.2f)' % rf_roc)
plt.plot(xg_fpr, xg_tpr, label='xgboost (area = %0.2f)' % xg_roc)
plt.plot(dt_fpr, dt_tpr, label='decision tree (area = %0.2f)' % dt_roc)
# Plot Base Rate ROC
plt.plot([0,1], [0,1],label='Base Rate' 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()





