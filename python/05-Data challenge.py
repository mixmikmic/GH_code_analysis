# some imports

# plotting, set up plotly offline
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
import plotly.graph_objs as go

# numpy
import numpy as np

# pandas
import pandas as pd

df_X_train = pd.read_csv('data/training_inputs.csv', sep=';')
df_X_test  = pd.read_csv('data/testing_inputs.csv', sep=';')
df_y_train = pd.read_csv('data/challenge_output_data_training_file_predict_which_clients_reduced_their_consumption.csv', 
                    sep=';')

n_train = df_X_train.shape[0]
print('Number of samples in train set: %s' % n_train)
df_X_train.head(10)

# print name of features
print('Available features are:')
print(list(df_X_train.keys()))

df_X_all = pd.concat([df_X_train, df_X_test])

features = ['C%s'% i for i in range(1, 20)]
print(features)

df_X_numerical = pd.get_dummies(df_X_all[features])

df_X_all[features].shape

df_X_numerical.shape

from sklearn import cross_validation, linear_model

# better cross-validation iterator than the default since it randomizes the samples.
# I will pass it to cross-validation functions
cv = cross_validation.ShuffleSplit(n_train)

cv_scores = cross_validation.cross_val_score(
    linear_model.LogisticRegression(), 
    df_X_numerical[:n_train], 
    df_y_train['TARGET'], scoring='roc_auc', cv=cv)

print('CV score: %s' % cv_scores.mean())

features = ['C%s'% i for i in range(1, 20)] + ['S%s'% i for i in range(1, 13)]
df_X_numerical = pd.get_dummies(df_X_all[features])
print(features)

from sklearn import pipeline, preprocessing

# using an imp
clf = pipeline.make_pipeline(preprocessing.Imputer(), linear_model.LogisticRegression())
cv_scores = cross_validation.cross_val_score(clf, 
    df_X_numerical[:n_train], 
    df_y_train['TARGET'], scoring='roc_auc', cv=cv)

print('CV score: %s' % cv_scores.mean())

from sklearn.learning_curve import validation_curve
# set some values for C
C_vals = np.logspace(-6, 6, 10)
clf = pipeline.make_pipeline(preprocessing.Imputer(), linear_model.LogisticRegression())

# if we want to sweep along the C parameter we need to specify that this belongs to the logistic regression
# part of the pipeline. This is done by specifying it with the name 'logisticregression__C'. The double
# underscore __ is used to denote "is a member of".
val_train, val_test = validation_curve(
    clf, df_X_numerical[:n_train], df_y_train['TARGET'], 'logisticregression__C', C_vals, cv=cv, 
    scoring='roc_auc')

data = [
    go.Scatter(
        x=C_vals,
        y=val_train.mean(1),
        error_y=dict(
            type='data',
            array=val_train.std(1),
            visible=True
        ),
        name='train error'
    ),
    go.Scatter(
        x=C_vals,
        y=val_test.mean(1),
        error_y=dict(
            type='data',
            array=val_train.std(1),
            visible=True
        ),
        name='validation error'
    )
]

layout = go.Layout(
    xaxis=dict(
        title='Polynomial degree',
        type='log',
    ),
    yaxis=dict(
        title='Score',
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)

features = ['C%s'% i for i in range(1, 20)] + ['S%s'% i for i in range(1, 13)] + ['Q%s'% i for i in range(1, 76)]
df_X_numerical = pd.get_dummies(df_X_all[features])
print(features)
print('Number of features: %s' % df_X_numerical.shape[1])

clf = pipeline.make_pipeline(preprocessing.Imputer(), linear_model.LogisticRegression())
cv_score = cross_validation.cross_val_score(clf, 
    df_X_numerical[:n_train], 
    df_y_train['TARGET'], scoring='roc_auc')
print('CV score: %s' % cv_score.mean())





