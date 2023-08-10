import pandas as pd
from constants import *

census_data = pd.read_csv('combined_data.csv')

X = census_data[feature_cols]
y = census_data['Democrat']

from sklearn.cross_validation import cross_val_score

# import the class
from sklearn.linear_model import LogisticRegression

# set the predictor and target variables
X = census_data[feature_cols]
y = census_data['Democrat']

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X, y)

# test accuracy of the model using 10-fold cross-validation
scores = cross_val_score(logreg, X, y, cv=20, scoring='roc_auc')
print(scores.mean())

# print logistic regression coefficients of each feature
coef = logreg.coef_[0]
zipped = zip(feature_cols, coef)

# predict the response for new observations

census_data['prediction'] = logreg.predict(X)
census_data.to_csv('census_data_with_predictions.csv')



