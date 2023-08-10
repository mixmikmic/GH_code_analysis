#we are going to be using the admissions data.
#admit column is binary, 1 means admitted, 0 means rejected. 
#indi. variables are gpa and gre
#Upload the data and take a look

import pandas as pd
admissions = pd.read_csv('admissions.csv')
admissions.head()

#Great, now lets make the initial logistic regressions model wiht the 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(admissions[['gpa']],admissions['admit'])
labels = model.predict(admissions[['gpa']])
admissions['predicted_label'] = labels
print(admissions['predicted_label'].value_counts())
print(admissions['admit'].value_counts())

# what fraction of the predictions were correct (actual lable matched predicted label?)
# accuracy = # of correctly predicted / # of observations

#duplicated the 'admit' column as 'actual_label' for easy 'actual vs prediction' comparison
admissions['actual_label'] = admissions['admit']

#compare the actual vs predicted. assign results to 'matches' column
admissions['matches']= admissions['actual_label'] == admissions['predicted_label']

#create 'correct predictions' dataframe to check if predicted and actual are actually the same
correct_predictions = admissions[admissions['matches']==True]
correct_predictions.head()

accuracy = len(correct_predictions) / len(admissions)
print(accuracy)

true_neg_filter =(admissions['predicted_label']==0) & (admissions['actual_label']==0)
true_pos_filter =(admissions['predicted_label']==1) & (admissions['actual_label']==1)
true_neg = len(admissions[true_neg_filter])
true_pos = len(admissions[true_pos_filter])
print(true_neg)
print(true_pos)

# From the previous screen
true_positive_filter = (admissions["predicted_label"] == 1) & (admissions["actual_label"] == 1)
true_positives = len(admissions[true_positive_filter])
false_negative_filter = (admissions["predicted_label"] == 0) & (admissions["actual_label"] == 1)
false_negatives = len(admissions[false_negative_filter])

sensitivity = true_positives / (true_positives + false_negatives)

print(sensitivity)

# From previous screens
true_positive_filter = (admissions["predicted_label"] == 1) & (admissions["actual_label"] == 1)
true_positives = len(admissions[true_positive_filter])
false_negative_filter = (admissions["predicted_label"] == 0) & (admissions["actual_label"] == 1)
false_negatives = len(admissions[false_negative_filter])

true_negative_filter = (admissions["predicted_label"] == 0) & (admissions["actual_label"] == 0)
true_negatives = len(admissions[true_negative_filter])

false_positive_filter = (admissions["predicted_label"] == 1) & (admissions["actual_label"] == 0)
false_positives = len(admissions[false_positive_filter])
specificity = (true_negatives) / (false_positives + true_negatives)
print(specificity)

