import pandas as pd
import numpy as np
import pickle

# Set Pandas display options so we can see more data
pd.set_option('display.width', 1000)

# Reload the trained model
tlo_classifier_file = "models/tlo_lr_classifier_02.18.16.dat"

logClassifier = pickle.load(open(tlo_classifier_file, "rb"))
print(logClassifier)

tlo_data_file = 'data/tlo_check_07_28_15_check_scores_anonymized.csv'
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

# Convert all strings to numerics
raw_data['failure_explanation'] = raw_data['failure_explanation'].apply(update_failure_explanations)
raw_data.head()

# Handle missing values
raw_data.fillna(0, inplace=True)
raw_data.head()

# Reorder the columns for splitting
# cols = list(raw_data)
# cols.insert(len(raw_data.columns)-1, cols.pop(cols.index('verified')))
# raw_data = raw_data.ix[:, cols]

cols = ['full_name_check_value',
        'ssn_score','dob_score',
        'n1_score',
        'n2_score',
        'n3_score',
        'n4_score',
        'n5_score',
        'n6_score',
        'n7_score',
        'n8_score',
        'n9_score',
        'n10_score',
        'n11_score',
        'n12_score',
        'n13_score',
        'n14_score',
        'ssn_match',
        'dob_match',
        'name_match',
        'failure_explanation',
        'last_name_check_value',
        'verified']
raw_data= raw_data[cols]
raw_data.head()

# Split the dataset between features and targets
tlo_data = raw_data.iloc[:,0:22].values
tlo_targets = raw_data['verified'].values

# tlo_data
# Make a prediction for each item in our data
for item in tlo_data:
    print(logClassifier.predict(item))



