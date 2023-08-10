# print_function for compatibility with Python 3
from __future__ import print_function

# NumPy for numerical computing
import numpy as np

# Pandas for DataFrames
import pandas as pd

# Matplotlib for visualization
import matplotlib.pyplot as plt

# display plots in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Seaborn for easier visualization
import seaborn as sns

# Scikit-Learn for Modeling
import sklearn

# Pickle for saving model files
import pickle

# Import Logistic Regression
from sklearn.linear_model import LogisticRegression

# Import RandomForestClassifier and GradientBoostingClassifer
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)

# Function for splitting training and test set
from sklearn.model_selection import train_test_split

# Function for creating model pipelines
from sklearn.pipeline import  make_pipeline

# For standardization
from sklearn.preprocessing  import StandardScaler

# Helper for cross-validation
from sklearn.model_selection import GridSearchCV

# Classification metrics (added later)
from sklearn.metrics import roc_curve, auc

# Load analytical base table from Module 2
df = pd.read_csv('analytical_base_table.csv')

# Create separate object for target variable
y = df['status']

# Create separate object for input features
X = df.drop('status', axis=1)

# Split X and y into train and test sets
X_train, X_test, y_train, y_test  = train_test_split(X, y,
                                                     test_size = 0.2,
                                                     random_state = 1234, 
                                                     stratify = df.status)

# Print number of observations in X_train, X_test, y_train, and y_test
print(len(X_train), len(X_test))
print(len(y_train), len(y_test))

# Pipeline dictionary
pipelines = {
    'l1': make_pipeline(StandardScaler(),
                        LogisticRegression(penalty='l1',random_state=123)),
    'l2': make_pipeline(StandardScaler(),
                        LogisticRegression(penalty='l2',random_state=123)),
    'rf': make_pipeline(StandardScaler(),
                        RandomForestClassifier(random_state=123)),
    'gb': make_pipeline(StandardScaler(),
                        GradientBoostingClassifier(random_state=123))
}

# List tuneable hyperparameters of our Logistic pipeline
pipelines['l1'].get_params()

# Logistic Regression hyperparameters
l1_hyperparameters = {
    'logisticregression__C': np.linspace(1e-3,1e3,10)
}
l2_hyperparameters = {
    'logisticregression__C': np.linspace(1e-3,1e3,10)
}

# Random Forest hyperparameters
rf_hyperparameters = {
    'randomforestclassifier__n_estimators': [100,200],
    'randomforestclassifier__max_features': ['auto','sqrt',0.33]
}

# Boosted Tree hyperparameters
gb_hyperparameters = {
    'gradientboostingclassifier__n_estimators': [100,200],
    'gradientboostingclassifier__learning_rate': [0.05, 0.1,0.2],
    'gradientboostingclassifier__max_depth': [1,3,5]
}

# Create hyperparameters dictionary
hyperparameters = {'l1': l1_hyperparameters,
                  'l2': l2_hyperparameters,
                  'rf': rf_hyperparameters,
                  'gb': gb_hyperparameters}

# Create empty dictionary called fitted_models
fitted_models = {}

# Loop through model pipelines, tuning each one and saving it to fitted_models
for name, pipeline in pipelines.items():
    # Create cross-validation object from pipeline and hyperparameters
    model = GridSearchCV(pipeline, hyperparameters[name], cv = 10, n_jobs=-1)
    
    # Fit model on X_train, y_train
    model.fit(X_train, y_train)
    
    # Store model in fitted_models[name] 
    fitted_models[name] = model
    
    # Print '{name} has been fitted'
    print('{0} has been fitted'.format(name))

# Display best_score_ for each fitted model
for name,model in fitted_models.items():
    print(name, "Score: ", model.best_score_)

# Classification metrics
from sklearn.metrics import roc_curve, auc

# Predict classes using L1-regularized logistic regression 
pred = fitted_models['l1'].predict(X_test)

# Display first 5 predictions
pred[:5]

# Import confusion_matrix
from sklearn.metrics import  confusion_matrix

# Display confusion matrix for y_test and pred
confusion_matrix(y_test, pred)

# Predict PROBABILITIES using L1-regularized logistic regression
pred = fitted_models['l1'].predict_proba(X_test)

# Get just the prediction for the positive class (1)
pred = [p[1] for p in pred]

# Display first 5 predictions
pred[:5]

# Calculate ROC curve from y_test and pred
fpr, tpr, thresholds = roc_curve(y_test, pred)

# Store fpr, tpr, thresholds in DataFrame and display last 10
pd.DataFrame({'FPR':fpr, 'TPR':tpr, 'Thresholds': thresholds}).tail(10)

# Initialize figure
fig = plt.figure(figsize=(8,8))
plt.title('Receiver Operating Characteristic')

# Plot ROC curve
plt.plot(fpr, tpr, label='l1')

# Diagonal 45 degree line
plt.plot([0,1],[0,1], 'k--')

# Axes limits and labels
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, pred)

# Calculate AUROC
print(auc(fpr,tpr))

# Code here

for name, model in fitted_models.items():
    pred = model.predict_proba(X_test)
    pred = [p[1] for p in pred]
    fpr, tpr, thresholds = roc_curve(y_test, pred)
    print(name, auc(fpr, tpr))

# Save winning model as final_model.pkl
with open('final_model.pkl', 'wb') as f:
    pickle.dump(fitted_models['rf'].best_estimator_, f)

