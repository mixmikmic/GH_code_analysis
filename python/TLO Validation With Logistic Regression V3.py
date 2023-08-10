# Import everything we need
import pandas as pd
import numpy as np

# Set Pandas display options so we can see more data
pd.set_option('display.width', 1000)

# Load the dataset
tlo_data_file = 'data/tlo_checks_07.28.15_cleaned.csv'

# Load the dataset into a pandas dataframe
raw_data = pd.DataFrame.from_csv(tlo_data_file, 
                       header=0, 
                       sep=',', 
                       index_col=0, 
                       parse_dates=True, 
                       encoding=None, 
                       tupleize_cols=False, 
                       infer_datetime_format=True)
raw_data.head()

# Lowercase the text fields
raw_data['failure_explanation'] = raw_data['failure_explanation'].str.lower()

# Failure Explanations: 'dob', 'name', 'ssn dob name', 'ssn', 'ssn name', 'ssn dob','dob name', nan
def update_failure_explanations(type):
    if type == 'dob':
        return 0
    elif type == 'name':
        return 1
    elif type == 'ssn dob name':
        return 2
    elif type == 'ssn':
        return 3
    elif type == 'ssn name':
        return 4
    elif type == 'ssn dob':
        return 5
    elif type == 'dob name':
        return 6

raw_data['failure_explanation'] = raw_data['failure_explanation'].apply(update_failure_explanations)
raw_data.head()

# Handle missing values
raw_data.fillna(0, inplace=True)
raw_data.head()

# Create two matrices for our model to use
tlo_data = raw_data.iloc[:,0:22].values
tlo_targets = raw_data['verified'].values

tlo_data

from sklearn import linear_model
logClassifier = linear_model.LogisticRegression(C=1, random_state=111)

from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(tlo_data, tlo_targets, test_size=0.20, random_state=111)
logClassifier.fit(X_train, y_train)

# Run the test data
predicted = logClassifier.predict(X_test)
predicted

# Evaluate the model
from sklearn import metrics
metrics.accuracy_score(y_test, predicted)

# Confusion matrix
metrics.confusion_matrix(y_test, predicted)

import pickle
tlo_classifier_file = "models/tlo_lr_classifier_02.18.16.dat"
pickle.dump(logClassifier, open(tlo_classifier_file, "wb"))

# Recreate it as a test
logClassifier2 = pickle.load(open(tlo_classifier_file, "rb"))
print(logClassifier2)



