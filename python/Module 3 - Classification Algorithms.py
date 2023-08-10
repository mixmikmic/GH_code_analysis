# print_function for compatibility with Python 3
from __future__ import print_function
# NumPy and Pandas
import numpy as np
import pandas as pd 

# Matplotlib, and remember to display plots in the notebook
from matplotlib import pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

# Seaborn for easier visualization
import seaborn as sns 

# Print unique classes for 'status' and the first 5 observations for 'status' in the raw dataset
raw_df = pd.read_csv('project_files/employee_data.csv')

print(raw_df.status.unique())
raw_df.status.head()

# Print unique classes for 'status' and the first 5 observations for 'status' in the analytical base table
abt_df = pd.read_csv('analytical_base_table.csv')

print(abt_df.status.unique())
abt_df.status.head()

# Input feature
x = np.linspace(0, 1, 100)
# Noise
np.random.seed(555)
noise = np.random.uniform(-0.2, 0.2, 100)

# Target variable
y = ((x + noise) > 0.5).astype(int)

# Reshape x into X
X = x.reshape(100, 1)

# Plot scatterplot of synthetic dataset
plt.scatter(X, y)

# Import LinearRegression and LogisticRegression
from sklearn.linear_model import LinearRegression, LogisticRegression

# Linear model
model = LinearRegression()
model.fit(X, y)

# Plot dataset and predictions
plt.scatter(X, y)
plt.plot(X, model.predict(X), 'k--')
plt.show()

# Logistic regression
model = LogisticRegression()
model.fit(X, y)

# predict()
model.predict(X)

# predict_proba()
pred = model.predict_proba(X[:10])

pred

# Class probabilities for first observation
pred[0]

# Positive class probability for first observation
pred[0][1]

# Just get the second value for each prediction
pred = [pred[1] for p in pred]

pred

# Logistic regression
model = LogisticRegression()
model.fit(X, y)

# Predict probabilities
pred = model.predict_proba(X)

# Just get the second value (positive class) for each prediction
pred = [p[1] for p in pred]

# Plot dataset and predictions
plt.scatter(X, y)
plt.plot(X, pred, 'k--')
plt.show()

def fit_and_plot_classifier(clf):
    # Fit model
    clf.fit(X, y)
    
    # Predict and take second value of each prediction
    pred = clf.predict_proba(X)
    pred = [p[1] for p in pred]
    
    # Plot
    plt.scatter(X, y)
    plt.plot(X, pred, 'k--')
    plt.show()
    
    # Return fitted model and predictions
    return clf, pred

# Logistic regression
clf, pred = fit_and_plot_classifier(LogisticRegression())

# More regularization
clf, pred = fit_and_plot_classifier(LogisticRegression(C=0.25))

# Less regularization
clf, pred = fit_and_plot_classifier(LogisticRegression(C=4))

# Basically no regularization
clf, pred = fit_and_plot_classifier(LogisticRegression(C=10000))

# L1 regularization
clf, pred = fit_and_plot_classifier(LogisticRegression)

# L1-regularized logistic regression
l1 = LogisticRegression(penalty='l1', random_state=123)

# L2-regularized logistic regression
l2 = LogisticRegression(penalty='12', random_state=123)

# L1 regularization with weaker penalty
clf, pred = fit_and_plot_classifier(LogisticRegression(penalty='l1', C=4))

# Import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# Random forest classifier
clf, pred = fit_and_plot_classifier(RandomForestClassifier(n_estimators=100))

# Import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Random forest classifier
clf, pred = fit_and_plot_classifier(GradientBoostingClassifier(n_estimators=100))

