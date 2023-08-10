import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import recall_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV
get_ipython().magic('matplotlib inline')

# Load and inspect the data
email_tbl = pd.read_csv('email/email_table.csv')
email_opened_tbl = pd.read_csv('email/email_opened_table.csv')
link_clicked_tbl = pd.read_csv('email/link_clicked_table.csv')

email_tbl.info()

email_opened_tbl.info()

link_clicked_tbl.info()

percent_opened = (email_tbl[email_tbl.email_id.isin(email_opened_tbl.email_id)].shape[0] / float(email_tbl.shape[0]))*100
percent_clicked = (email_tbl[email_tbl.email_id.isin(link_clicked_tbl.email_id)].shape[0] / float(email_tbl.shape[0]))*100
print "Percent opened {0}".format(percent_opened)
print "Percent clicked {0}".format(percent_clicked)

# Create a boxplot of past user purchases
plt.boxplot(email_tbl.user_past_purchases)

# Remove outliers
non_outliers = email_tbl[email_tbl.user_past_purchases < 12]

# Add a 'link_clicked' column to the dataframe
clicked_emails = set(link_clicked_tbl.email_id)
non_outliers['link_clicked'] = non_outliers.email_id.apply((lambda x: 1 if x in clicked_emails else 0))

non_outliers.head()

non_outliers.user_country.value_counts(normalize=True)

# Over 90% of the user base is in the US and UK. For now, we'll simplify the problem and focus only on users in the US and UK.
us_uk_emails = non_outliers[non_outliers.user_country.isin(['US','UK'])]

# Create a binary column 'country_US' to indiciate if the user is in the US or UK
us_uk_emails['country_US'] = us_uk_emails.user_country.apply((lambda x: 1 if x == 'US' else 0))

# Create a binary column for the email text. Value is 1 for short_emails, 0 for long emails
us_uk_emails['is_short'] = us_uk_emails.email_text.apply((lambda x: 1 if x == 'short_email' else 0))

# Create a binary column for email version. Value is 1 for generic emails, 0 for personalized emails
us_uk_emails['is_generic'] = us_uk_emails.email_version.apply((lambda x: 1 if x == 'generic' else 0))

us_uk_emails[us_uk_emails['link_clicked'] == 1].hour.value_counts(normalize=True)

# Very few people who click on the email links do so during the evening hours. We'll therefore lump these variables into a
# a category called EVENING to reduce the number of dummy variables to create for the model.
evening_hours = [19,20,21,22,23,24]
us_uk_emails['hour_sent'] = us_uk_emails.hour.apply((lambda x: 'EVENING' if x in evening_hours else str(x)))

# Create dummy variables for the weekday and hour columns
weekday_dummies = pd.get_dummies(us_uk_emails.weekday,prefix='weekday').drop('weekday_Sunday',axis=1)
hour_dummies = pd.get_dummies(us_uk_emails.hour_sent,prefix='hour_sent').drop('hour_sent_EVENING',axis=1)

# Create the feature matrix
df = us_uk_emails[['user_past_purchases','is_short','is_generic','link_clicked']]
df = df.join(weekday_dummies)
df = df.join(hour_dummies)
y = df.pop('link_clicked').values
X = df.values

'''
Since a very tiny percentage of users who clicked the link in the email, we will undersample the majority class to balance
the dataset.
'''
def undersample(X,y):
    majority_X = X[y == 0]
    majority_y = y[y == 0]
    minority_y = y[y == 1]
    minority_X = X[y == 1]
    minority_size = y[y == 1].shape[0]
    
    inds = np.arange(minority_size)
    samples = np.random.choice(inds, minority_size,replace=False)
    
    undersampled_X = np.concatenate((majority_X[samples],minority_X), axis=0)
    undersampled_y = np.concatenate((majority_y[samples],minority_y), axis=0)
    return (undersampled_X, undersampled_y)

# Split the data into a training set and validation set
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.1)

# Undersample the training data to balance the classes
X_sub,y_sub = undersample(X_train,y_train)

# Train and tune a random forest classifer 
rf_params = {'n_estimators':[20,30,40,50],
             'criterion':['gini','entropy'],
             'max_features':['auto','log2'],
             'min_samples_leaf':[1,2,5,10,11,12,13,14,15],
             'min_samples_split':[2, 3, 5, 7,10]}
rf_clf = GridSearchCV(RandomForestClassifier(),rf_params, scoring = 'roc_auc')
rf_clf.fit(X_sub,y_sub)

rf_clf.best_params_

rf_clf.best_score_

# Train and tune a logistic regression classifer 
lr_params = {'C':[0.1,0.2, 0.3, 0.4]}
lr_clf = GridSearchCV(LogisticRegression(),lr_params,scoring='roc_auc')
lr_clf.fit(X_sub,y_sub)

lr_clf.best_params_

lr_clf.best_score_

# Train and tune a Gradient Boosting classifer 
gbc_params = {'learning_rate':[0.1,0.2,0.3,0.4],
              'n_estimators':[100,200],
              'min_samples_leaf':[1,5,7,8],
              'min_samples_split':[2, 3, 5, 7,10],
              'max_features':['auto','log2']
             }
gbc_clf = GridSearchCV(GradientBoostingClassifier(),gbc_params,scoring='roc_auc')
gbc_clf.fit(X_sub,y_sub)

gbc_clf.best_params_

gbc_clf.best_score_

# Train and tune a Support Vector classifer 
svc_params = {'kernel':['rbf','sigmoid','linear','poly'],
              'C':[0.1,0.5,1,5,10],
              'degree':[1,3,5],
              'coef0':[0.5,0.6,0.7]
             }
svc_clf = GridSearchCV(SVC(),svc_params,scoring='roc_auc')
svc_clf.fit(X_sub,y_sub)

svc_clf.best_params_

svc_clf.best_score_

# Test models on the validation set
lr = LogisticRegression(C=0.1)
lr.fit(X_sub,y_sub)
lr_pred = lr.predict(X_val)
lr_fpr, lr_tpr, lr_thresholds = roc_curve(y_val, lr_pred)
lr_recall = recall_score(y_val, lr_pred)
lr_auc = auc(lr_fpr, lr_tpr)

rf = RandomForestClassifier(criterion = 'entropy',
                            max_features = 'auto',
                            min_samples_leaf = 12,
                            min_samples_split = 7,
                            n_estimators= 40)
rf.fit(X_sub,y_sub)
rf_pred = rf.predict(X_val)
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_val, rf_pred)
rf_recall = recall_score(y_val, rf_pred)
rf_auc = auc(rf_fpr, rf_tpr)

gbc = GradientBoostingClassifier(learning_rate= 0.1,
                                 max_features ='log2',
                                 min_samples_leaf = 5,
                                 min_samples_split = 2,
                                 n_estimators =100)
gbc.fit(X_sub,y_sub)
gbc_pred = gbc.predict(X_val)
gbc_fpr, gbc_tpr, gbc_thresholds = roc_curve(y_val, gbc_pred)
gbc_recall = recall_score(y_val, gbc_pred)
gbc_auc = auc(gbc_fpr, gbc_tpr)

svc = SVC(C=1, coef0=0.6, degree=1,kernel='poly')
svc.fit(X_sub,y_sub)
svc_pred = svc.predict(X_val)
svc_fpr, svc_tpr, svc_thresholds = roc_curve(y_val, svc_pred)
svc_recall = recall_score(y_val, svc_pred)
svc_auc = auc(svc_fpr, svc_tpr)

print "Logistic Regression (AUC, Recall) {0}".format((lr_auc,lr_recall))
print "Random Forest (AUC, Recall) {0}".format((rf_auc,rf_recall))
print "Gradient Boosting Classifier (AUC, Recall) {0}".format((gbc_auc,gbc_recall))
print "Support Vector Classifier (AUC, Recall) {0}".format((svc_auc,svc_recall))

# THe Random Forest classifer appears to be the best. Let's check out it's confusion matrix.
confusion_matrix(y_val,rf_pred)

def plot_hourly_counts(ax, df, atitle):
    email_hours = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    email_counts = df.groupby(['hour','link_clicked'])['email_id'].count().to_dict()

    counts_clicked = []
    counts_noclicked = []
    bar_width = 0.35
    ind = np.arange(len(email_hours))

    for e in email_hours:
        if (e,1) in email_counts:
            counts_clicked.append(email_counts[(e,1)])
        else:
            counts_clicked.append(0)
        if (e,0) in email_counts:
            counts_noclicked.append(email_counts[(e,0)])
        else:
            counts_noclicked.append(0)
    ax.bar(ind,counts_clicked,bar_width,color='r',label='clicked')
    ax.bar(ind+bar_width,counts_noclicked,bar_width,color='g',label='not clicked')
    ax.set_title(atitle)

fig, axes = plt.subplots(7,1)
fig.set_size_inches(18.5, 10.5)
dow = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
pos = 0
for i in range(7):
    df = non_outliers[non_outliers.weekday ==  dow[pos]]
    plot_hourly_counts(axes[i], df, dow[pos])
    pos += 1
plt.legend()

fig, axes = plt.subplots(7,1)
fig.set_size_inches(18.5, 10.5)
dow = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
pos = 0
us_emails = non_outliers[non_outliers.user_country == 'US']
for i in range(7):
    df = us_emails[us_emails.weekday ==  dow[pos]]
    plot_hourly_counts(axes[i], df, dow[pos])
    pos += 1
plt.legend()

fig, axes = plt.subplots(7,1)
fig.set_size_inches(18.5, 10.5)
dow = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
pos = 0
us_emails = non_outliers[non_outliers.user_country == 'UK']
for i in range(7):
    df = us_emails[us_emails.weekday ==  dow[pos]]
    plot_hourly_counts(axes[i], df, dow[pos])
    pos += 1
plt.legend()

fig, axes = plt.subplots(7,1)
fig.set_size_inches(18.5, 10.5)
dow = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
pos = 0
us_emails = non_outliers[non_outliers.user_country == 'ES']
for i in range(7):
    df = us_emails[us_emails.weekday ==  dow[pos]]
    plot_hourly_counts(axes[i], df, dow[pos])
    pos += 1
plt.legend()

fig, axes = plt.subplots(7,1)
fig.set_size_inches(18.5, 10.5)
dow = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
pos = 0
us_emails = non_outliers[non_outliers.user_country == 'FR']
for i in range(7):
    df = us_emails[us_emails.weekday ==  dow[pos]]
    plot_hourly_counts(axes[i], df, dow[pos])
    pos += 1
plt.legend()



