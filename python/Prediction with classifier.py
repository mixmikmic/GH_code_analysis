get_ipython().magic('matplotlib inline')
import pandas as pd
from datetime import datetime
import numpy as np
import seaborn as sns
import pickle
#import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import xgboost
from model.models import TradeModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.grid_search import GridSearchCV
from pandas_ml import ConfusionMatrix
from service.files_service import get_files

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

all_files = get_files(folder = 'data/all_data/resampled_D/',extension='.csv',as_dict=True,filter_on='_D_')

def make_prediction(tm,date):
    df = tm.df.copy()
   
    
    predictors =df.columns.tolist()
    df['target']=((df.High-df.High.shift(-1))*10000>=10)*1
    
    clf = xgboost.XGBClassifier()
    
    df_train = df[df.index<date]
    df_test = df[df.index==date]
 
    
      
    X_train,y_train = df_train[predictors].values,df_train.target.values
    X_test, y_test  = df_test[predictors].values,df_test.target.values
    
    clf = cv_optimize(clf,{},X_train,y_train)
    
    df_test['prediction'] = clf.predict(X_test)
    
    score =  clf.score(X_test,y_test)

    
    return y_test[0],df_test['prediction'][0],clf

def cv_optimize(clf, parameters, Xtrain, ytrain, n_folds=5):
    gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds, n_jobs=-1)
    gs.fit(Xtrain, ytrain)
    
    return gs.best_estimator_    

def compute_confusion_matrix(tm):
    real_result = []
    predicted_result=[]
    
    test_sample=100
    for dt in (tm.df.tail(test_sample).index.values):
        real,predicted,_ = make_prediction(tm, dt)
        break
        
        real_result.append(real)
        predicted_result.append(predicted)
        
    return ConfusionMatrix(real_result, predicted_result)

confusion_matrixes = {}
for k,v in all_files.iteritems():
    tm =TradeModel('data/all_data/resampled_D/'+v, name=k, datetime_col='ds')
    confusion_matrixes[k]=compute_confusion_matrix(tm)

confusion_matrixes['resampled_D_BRENTCMDUSD']

pickle.dump(confusion_matrixes, open('data/confusion_matrix2.p', "wb"))

cmo = pickle.load(open('data/confusion_matrix2.p', "rb"))

df_CM_all = pd.DataFrame()
for k,v in cmo.iteritems():
    print v.stats_class.loc['PPV: Pos Pred Value (Precision)'][1]
    
    df_cm = cmo[k].to_dataframe()
    df_cm.index=[k +'_Actual_Negative',k +'_Actual_Positive']
    df_cm.columns=['Predicted_negative','Predicted_Positive']
    df_CM_all = pd.concat([df_CM_all,df_cm],axis=0)

from datetime import timedelta
datetime(2016,12,8) - timedelta(100)

def testo(row):
    if  ('Positive' in row.name):
        return 1
    elif ('Negative' in row.name):
        return -1
    else :
        return 0

df_CM_all['win_lose']=df_CM_all.apply(testo,axis=1)

df_CM_all['gain_loss']=df_CM_all.win_lose * (100) * df_CM_all.Predicted_Positive
df_CM_all['invested']=df_CM_all.Predicted_Positive * (100)

df_CM_all

print df_CM_all.gain_loss.sum()
print df_CM_all.invested.sum()
print df_CM_all.gain_loss.sum()*1.0/df_CM_all.invested.sum()*100.0

def make_roc(name, clf, ytest, xtest, ax=None, labe=50, proba=True, skip=0):
    initial=False
    if not ax:
        ax=plt.gca()
        initial=True
    if proba:
        fpr, tpr, thresholds=roc_curve(ytest, clf.predict_proba(xtest)[:,1])
    else:
        fpr, tpr, thresholds=roc_curve(ytest, clf.decision_function(xtest))
    roc_auc = auc(fpr, tpr)
    if skip:
        l=fpr.shape[0]
        ax.plot(fpr[0:l:skip], tpr[0:l:skip], '.-', alpha=0.3, label='ROC curve for %s (area = %0.2f)' % (name, roc_auc))
    else:
        ax.plot(fpr, tpr, '.-', alpha=0.3, label='ROC curve for %s (area = %0.2f)' % (name, roc_auc))
    label_kwargs = {}
    label_kwargs['bbox'] = dict(
        boxstyle='round,pad=0.3', alpha=0.2,
    )
    for k in xrange(0, fpr.shape[0],labe):
        #from https://gist.github.com/podshumok/c1d1c9394335d86255b8
        threshold = str(np.round(thresholds[k], 2))
        ax.annotate(threshold, (fpr[k], tpr[k]), **label_kwargs)
    if initial:
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC')
    ax.legend(loc="lower right")
    return ax

def test_ROC(date):
    tm =TradeModel('data/all_data/resampled_D/resampled_H_USDCAD_UTC_1 Min_Bid_2010.01.02_2016.12.08.csv',                   name='test', datetime_col='ds')
    df=tm.df.copy()
    
    predictors =df.columns.tolist()
    
    df['target']=((df.High-df.High.shift(-1))*10000>=10)*1
    
    clf = xgboost.XGBClassifier()
    
    df_train = df[df.index<date]
    df_test = df[df.index>=date]
 
    print df_test.shape
      
    X_train,y_train = df_train[predictors].values,df_train.target.values
    X_test, y_test  = df_test[predictors].values,df_test.target.values
    
    clf.fit(X_train,y_train)
    
    prediction = clf.predict_proba(X_test)
    
    return X_test, y_test,clf
    

X_test, y_test,clf = test_ROC(datetime(2016,12,8) - timedelta(100))

y_test

ax = make_roc('test',clf,y_test,X_test)

