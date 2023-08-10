import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# scores
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


#define category columns
cat_cols = {
    'grade': 'category',
    'sub_grade': 'category',
    'home_ownership': 'category',
    'verification_status': 'category',
    'loan_status': 'category',
    'purpose':'category',
    'addr_state':'category',
    'initial_list_status':'category',
    'inq_last_6mths_cat': 'category',
    'pub_rec_cat':'category',
    'fully_funded':'category'
}

# import data
df_raw = pd.read_csv('../data/processed/LCData_processed.csv', dtype=cat_cols, index_col=0)

# have a look at columns with missing data
df_raw.info()

# drop null data
df_processed = df_raw[pd.notnull(df_raw.revol_util)]

# drop null data
df_processed = df_processed[pd.notnull(df_processed.collections_12_mths_ex_med)]

# remove 3 columns
df_processed = df_processed.drop(['tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim'], axis=1)

df_processed.info()

# create a list of my engineered feature names
feat_eng_cols = ['low_deliquences_jobs',
                 'issue_month', 
                 'issue_year', 
                 'inq_last_6mths_cat',
                 'pub_rec_cat',
                 'fully_funded'
                 ]

# We are trying to determine whether or not a loan has 'defaulted'.  
# As a result, we only want to look at loans that are complete, meaning 
# we drop loans that are current, or haven't defaulted fully.  We only look at loans within the below status'
status = ['Fully Paid', 
          'Charged Off', 
          'Default', 
          'Does not meet the credit policy. Status:Fully Paid',
          'Does not meet the credit policy. Status:Charged Off']

# filter out loans
df = df_processed[df_processed.loan_status.str.contains('|'.join(status))]

sc = StandardScaler()

# create our X and y datasets
X = df[feat_eng_cols]
y = df.default_status

# create the dummies for our category columns
X = pd.get_dummies(X)
X = sc.fit_transform(X)

# split our data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# fit the classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# how is our class order defined? 
clf.classes_

# predict X
y_pred = clf.predict(X_test)

# find the assigned scores for the y predictions (which is the second column in predicitions)
y_score = clf.predict_proba(X_test)[:,1]

# find the scores
r2_score_test = clf.score(X_test, y_test)
mse_test = mean_squared_error(y_test,y_pred)
roc_test = roc_auc_score(y_test, y_score)

# print scores
print('r2 test: {}'.format(r2_score_test))
print('mse: {}'.format(mse_test))
print('roc: {}'.format(roc_test))

# plot the ROC curve
plt.figure(figsize=(6,6))

# calculate false positives & true positives
fpr, tpr, thresholds = roc_curve(y_test, y_score)

# plot ROC curve & 50% line
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], color='black')

# set labels
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('ROC curve - Logistic Regression')

plt.show()

# create target_names
t_names = ['0', '1']

print(classification_report(y_test, y_pred, target_names=t_names))

# create confusion matrix
confusion_matrix(y_test, y_pred)

# let's find the cross validation score
scores = cross_val_score(clf, X, y, cv=5)
scores

print("Accuracy: {0:.2f} (+/- {1:.5f})".format(scores.mean(), scores.std() * 2))





# Lets calculate the default rates for loans based on lending club's grades
for g in sorted(df.grade.unique()):
    
    #calculate default rate
    default_rate = df[(df.default_status ==1) & (df.grade==g)].loan_amnt.count() / df[df.grade==g].loan_amnt.count() 

    print('{0}: {1:.3f}'.format(g, default_rate))

# Lets calculate the expected default rates for loans by grade using our classifier 

grade_scores = {}

# loop through each grade
for g in sorted(df.grade.unique()):
    df_grades = df[df.grade==g]

    # create our X and y datasets
    X = df_grades[feat_eng_cols]
    y = df_grades.default_status

    # create the dummies for our category columns & scale
    X = pd.get_dummies(X)
    X = sc.fit_transform(X)

    # split our data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # fit the classifier
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # find the scores
    print('{0}: {1:.3f}'.format(g, np.mean(clf.predict_proba(X_train)[:, 1])))

# Using only the engineered features did not go very well - our precision and recall scores 
# for the positive class were zero. We need additional features to help us improve our model 
# performance.  Let's consider using ALL features for a model this time around.

# I know that there are several columns which are heavily correlated with the default status - so I will 
# remove these columns to see how the model does without them.  In particular, we will drop:
    
drop_cols = ['grade',
            'sub_grade',
            'loan_status',
            'int_rate',
            'default_status']

# create our X and y datasets
X = df.drop(drop_cols, axis=1)
y = df.default_status

# create the dummies for our category columns & scale
X = pd.get_dummies(X)
X = sc.fit_transform(X)

# split our data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# fit the classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# how is our class order defined? 
clf.classes_

# predict X
y_pred = clf.predict(X_test)

# find the assigned scores for the y predictions (which is the second column in predicitions)
y_score = clf.predict_proba(X_test)[:,1]

# find the scores
r2_score_test = clf.score(X_test, y_test)
mse_test = mean_squared_error(y_test,y_pred)
roc_test = roc_auc_score(y_test, y_score)

# print scores
print('r2 test: {}'.format(r2_score_test))
print('mse: {}'.format(mse_test))
print('roc: {}'.format(roc_test))

# plot the ROC curve
plt.figure(figsize=(6,6))

# calculate false positives & true positives
fpr, tpr, thresholds = roc_curve(y_test, y_score)

# plot ROC curve & 50% line
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], color='black')

# set labels
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('ROC curve - Logistic Regression')

plt.show()

# create target_names
t_names = ['0', '1']

print(classification_report(y_test, y_pred, target_names=t_names))





