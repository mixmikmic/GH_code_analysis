import pickle
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from sklearn.preprocessing import LabelEncoder, FunctionTransformer, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

import os
import sys

module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

from nba.database import Database

plt.rc('font', size=14)        # controls default text sizes
plt.rc('axes', titlesize=16)   # fontsize of the axes title
plt.rc('axes', labelsize=16)   # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)  # fontsize of the tick labels
plt.rc('ytick', labelsize=14)  # fontsize of the tick labels
plt.rc('legend', fontsize=12)  # legend fontsize

database = Database('../nba.db')
season_stats = database.season_stats()

conn = sqlite3.connect('../nba.db')
games = pd.read_sql('SELECT * FROM games', conn)

games[[x is None for x in games.HOME_WL]]

games = games[[x is not None for x in games.HOME_WL]]

seasons = season_stats.filter(regex='SEASON|TEAM')
games = games.merge(seasons, left_on=['SEASON', 'HOME_TEAM_ID'], right_on=['SEASON', 'TEAM_ID'])
games = games.merge(seasons, left_on=['SEASON', 'AWAY_TEAM_ID'], right_on=['SEASON', 'TEAM_ID'],
                    suffixes=('', '_AWAY'))
games = games[games.SEASON >= 1990]

stats = ['FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'AST', 'OREB', 'DREB', 'REB', 'STL', 'TOV', 'BLK',
         'EFG', 'TOV_PCT', 'OREB_PCT', 'DREB_PCT', 'FT_PER_FGA', 'FOUR_FACTORS',
         'SRS', 'PLUS_MINUS', 'OFF_RTG', 'DEF_RTG', 'NET_RTG']
labels = ['TEAM_' + s for s in stats]
labels.extend(['TEAM_' + s + '_AWAY' for s in stats])

# Select lables from the games DataFrame and turn the HOME_WL column to a numeric column
X, y = games[labels], LabelEncoder().fit_transform(games.HOME_WL)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

print "Training set home win percentage: {:.2f}%".format(np.mean(y_train) * 100)
print "Test set home win percentage: {:.2f}%".format(np.mean(y_test) * 100)

f = open('databall.pkl', 'wb')
pickle.dump([X, X_train, X_test, y, y_train, y_test], f)
f.close()

model = SelectKBest(f_classif, k=10).fit(X_train, y_train)

for label in X_train.columns[model.get_support()]:
    print label

model = SelectKBest(chi2, k=10).fit(MinMaxScaler((0, 1)).fit_transform(X_train), y_train)

for label in X_train.columns[model.get_support()]:
    print label

def cross_val_roc_curve(model, X, y, ax, k=10, label='Mean', show_folds=False):
    # Compute cross-validated ROC curve and area under the curve
    kfold = StratifiedKFold(n_splits=k)
    proba = cross_val_predict(model, X, y, cv=kfold, method='predict_proba')
    mean_fpr, mean_tpr, thresholds = roc_curve(y, proba[:, 1])
    mean_auc = roc_auc_score(y, proba[:, 1])
    
    # Loop over k folds and get ROC curve for each fold
    if show_folds:
        for i, (train, test) in enumerate(kfold.split(X, y)):
            # Fit model for the ith fold
            proba = model.fit(X.iloc[train], y[train]).predict_proba(X.iloc[test])
            
            # Compute ROC curve and area under the curve
            fpr, tpr, thresholds = roc_curve(y[test], proba[:, 1])
            roc_auc = roc_auc_score(y[test], proba[:, 1])
            ax.plot(fpr, tpr, lw=1, label='Fold %d (Area = %0.2f)' % (i+1, roc_auc))
    
        ax.plot(mean_fpr, mean_tpr, 'k--', label='%s (Area = %0.2f)' % (label, mean_auc), lw=2)
    else:
        ax.plot(mean_fpr, mean_tpr, label='%s (Area = %0.2f)' % (label, mean_auc), lw=2)

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

def cross_val_precision_recall_curve(model, X, y, ax, k=10, label='Mean', show_folds=False):
    # Compute cross-validated precision/recall curve and area under the curve
    kfold = StratifiedKFold(n_splits=k)
    proba = cross_val_predict(model, X, y, cv=kfold, method='predict_proba')
    mean_precision, mean_recall, thresholds = precision_recall_curve(y, proba[:, 1])
    mean_auc = average_precision_score(y, proba[:, 1])
    
    # Loop over k folds and get precision/recall curve for each fold
    if show_folds:
        for i, (train, test) in enumerate(kfold.split(X, y)):
            # Fit model for the ith fold
            proba = model.fit(X.iloc[train], y[train]).predict_proba(X.iloc[test])
            
            # Compute precision/recall curve and area under the curve
            precision, recall, thresholds = precision_recall_curve(y[test], proba[:, 1])
            pr_auc = average_precision_score(y[test], proba[:, 1])
            ax.plot(recall, precision, lw=1, label='Fold %d (Area = %0.2f)' % (i+1, pr_auc))
    
        ax.plot(mean_recall, mean_precision, 'k--', label='%s (Area = %0.2f)' % (label, mean_auc), lw=2)
    else:
        ax.plot(mean_recall, mean_precision, label='%s (Area = %0.2f)' % (label, mean_auc), lw=2)
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')

def cross_val_curves(model, X, y, k=10):
    plt.figure(figsize=(16, 6))
    
    # Plot ROC curve
    ax = plt.subplot(121)
    cross_val_roc_curve(model, X, y, ax, k=k, label='Mean', show_folds=True)
    ax.legend()
    
    # Plot precision/recall curve
    ax = plt.subplot(122)
    cross_val_precision_recall_curve(model, X, y, ax, k=k, label='Mean', show_folds=True)
    ax.legend()
    ax.set_ylim(0.4)
    
    plt.show()

def select_columns(df, names, columns=X_train.columns):
    return df[:, [i for i, col in enumerate(columns) if any(n in col for n in names)]]

attributes = ['TEAM_SRS', 'TEAM_SRS_AWAY']
model = LogisticRegression()
cross_val_curves(model, X_train[attributes], y_train)

# Define four factors
four_factors = ['EFG', 'TOV_PCT', 'OREB_PCT', 'DREB_PCT', 'FT_PER_FGA']
    
# Make a pipeline that selects the desired attributes prior to calling classifier
selector = FunctionTransformer(partial(select_columns, names=four_factors))
model = make_pipeline(selector, LogisticRegression())
cross_val_curves(model, X_train, y_train)

attributes = ['TEAM_FOUR_FACTORS', 'TEAM_FOUR_FACTORS_AWAY']
model = LogisticRegression()
cross_val_curves(model, X_train[attributes], y_train)

# Make pipeline that selects attributes prior to calling the classifier
selector = SelectKBest(f_classif, k=10)
model = make_pipeline(selector, LogisticRegression())
cross_val_curves(model, X_train, y_train)

# Define attributes to use for each curve
attributes = [['SRS'], ['PLUS_MINUS'], ['NET_RTG'], ['OFF_RTG', 'DEF_RTG'],
              four_factors, ['FOUR_FACTORS']]
labels = ['SRS', 'Plus/Minus', 'Net Rtg', 'Off/Def Rtg', 'Four Factors', 'Weighted Four Factors']

plt.figure(figsize=(16, 6))
ax = plt.subplot(121)

# Plot ROC curves for each set of attributes
for (a, l) in zip(attributes, labels):
    # Make transformer that selects the desired attributes from the DataFrame
    selector = FunctionTransformer(partial(select_columns, names=a))
    
    # Make a pipeline that selects the desired attributes prior to the classifier
    model = make_pipeline(selector, LogisticRegression())
    cross_val_roc_curve(model, X_train, y_train, ax, label=l)

ax.legend(fontsize=14)
ax = plt.subplot(122)

# Plot ROC curves for each set of attributes
for (a, l) in zip(attributes, labels):
    # Make transformer that selects the desired attributes from the DataFrame
    selector = FunctionTransformer(partial(select_columns, names=a))
    
    # Make a pipeline that selects the desired attributes prior to the classifier
    model = make_pipeline(selector, LogisticRegression())
    cross_val_precision_recall_curve(model, X_train, y_train, ax, label=l)
    
ax.legend(fontsize=14)
plt.show()

