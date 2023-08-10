import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import RandomizedLogisticRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

# Loading the data saved from the last notebook
X = np.load('./_data/madelon_db.p')

X.shape

y = X[1001].as_matrix(columns=None)
y

cols = list(range(0, 1001, 1))
X = X[cols]
X = X.as_matrix(columns=None)
X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Instatiate and fit the logistic regression model
logr = LogisticRegression()
logr.fit(X_train,y_train)

# Threshold chosen from earlier testing
threshold = 0.04

stability_selection = RandomizedLogisticRegression(n_resampling=300,
                                                   n_jobs=1,
                                                   random_state=101,
                                                   scaling=0.15,
                                                   sample_fraction=0.50,
                                                   selection_threshold=threshold)

interactions = PolynomialFeatures(degree=4, interaction_only=True)

model = make_pipeline(stability_selection, interactions, logr)

model.fit(X_train, y_train)

print('Number of features picked by stability selection: %i' % np.sum(model.steps[0][1].all_scores_ >= threshold))

print('Area Under the Curve: %0.5f' % roc_auc_score(y_val, model.predict_proba(X_val)[:,1]))

feature_filter = model.steps[0][1].all_scores_ >= threshold

counter = -1
important_features = []
for i in feature_filter:
    counter += 1
    if i == True:
        important_features.append(counter)
print('Number of important features:', len(important_features))
print('List of important features:', important_features)



