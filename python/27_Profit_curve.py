from __future__ import division

from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.svm import SVC

from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

df_source = pd.read_csv('data/churn_sample.csv')

df_source[:2].T

dropcols = ['Phone', 'Area Code', 'State']
df = df_source.drop(dropcols, axis=1)

df['Churn?'] = df['Churn?'].map({'False.':False, "True.":True})
df["Int'l Plan"]=df["Int'l Plan"].map({'yes': True,'no':False})
df["VMail Plan"] = df["VMail Plan"].map({'yes': True,'no':False})

def cmatrix(y_truth, y_prediction):
    y_truth, y_prediction = np.array(y_truth), np.array(y_prediction)
    
    tp = np.sum((y_truth == y_prediction) & (y_truth == 1))
    tn = np.sum((y_truth == y_prediction) & (y_truth == 0))
    fp = np.sum((y_truth != y_prediction) & (y_prediction == 1))
    fn = np.sum((y_truth != y_prediction) & (y_prediction == 0))

    
    return np.array([[tp, fp], [fn, tn]])
    

y_true = [1, 1, 1, 1, 1, 1, 0]
y_predict = [1, 1, 1, 1, 0, 1, 0]

cmatrix(y_true, y_predict)

# sklearn confusion matrix
confusion_matrix(y_true, y_predict)

# In order to obtain this format ([[tp, fp], [fn, tn]]) we need to add .T
confusion_matrix(y_true,y_predict, labels=[1,0]).T

def profit_curve(cost_benefit, predicted_probabilities, labels):
    profits = []
    percentages = []
    sorted_probabilities = sorted(predicted_probabilities, reverse=True)
    for threshold in sorted_probabilities:
        predicting = (predicted_probabilities > threshold).astype(int)
        cm = confusion_matrix(labels, predicting, labels=[1,0]).T
        profits.append(np.sum(cost_benefit * cm)* 1. / len(labels))
        percentages.append(cm.sum(axis=1)[0]/cm.sum())
    return profits, percentages

probas = np.array([0.2, 0.6, 0.4])
labels = np.array([0, 0, 1])
cb = np.array([[6, -3], [0, 0]])

profit_curve(cb, probas, labels)

y = df.pop('Churn?').values
feature_names = df.columns
X = df.values

X_train, X_test, y_train, y_test = train_test_split(X, y)

def plot_profit_curve(model, cost_benefit, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    model_name = model.__class__.__name__.replace('Regressor', '')
    
    y_predicted_probabilities = model.predict_proba(X_test)[:,1]
    
    profits, percentages = profit_curve(cost_benefit, y_predicted_probabilities, y_test)
    
    plt.plot(percentages, profits, label=model_name)
    plt.xlabel('Percentage predicted as positive')
    plt.ylabel('profit')
    plt.legend(loc='lower right')
    
    return profits, percentages

def plot_profit_curve_(model, cost_benefit, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    model_name = model.__class__.__name__.replace('Regressor', '')
    
    y_predicted_probabilities = model.predict_proba(X_test)[:,1]
    
    profits, percentages = profit_curve(cost_benefit, y_predicted_probabilities, y_test)
    
    plt.plot(percentages, profits, label=model_name)
    plt.xlabel('Percentage predicted as positive')
    plt.ylabel('profit')
    plt.legend(loc='lower right')
    

cost_benefit_ = np.array([[6, -3], [0, 0]])

plot_profit_curve_(LogisticRegression(),cost_benefit_, X_train, X_test, y_train, y_test)

models = [RF(), LR(), GBC(), SVC(probability=True)]

for model in models:
    profits, percentages = plot_profit_curve(model, cost_benefit_, X_train, X_test, y_train, y_test)
    print 'Max Profit {}: {}'.format(model.__class__.__name__, np.max(profits))
    print 'At Percentage {}: {}'.format(model.__class__.__name__, percentages[np.argmax(np.array(profits))])
    print '------'
    
plt.axhline(.9, ls='--')
plt.axhline(0, ls='--')
plt.title('Profit Curves')
plt.xlabel('Percentage of test intances (decresing by score)')
plt.ylabel('Profit')
plt.legend(loc='center left', bbox_to_anchor=[1, .5])
plt.show()









