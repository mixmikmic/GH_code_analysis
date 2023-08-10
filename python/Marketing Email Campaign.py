# import libraries
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
get_ipython().magic('matplotlib inline')
import xgboost as xgb
from sklearn.feature_selection import chi2,f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,roc_curve,auc,precision_score,recall_score,precision_recall_curve
from sklearn.cross_validation import train_test_split

seed = 9999

# Load email data/info from csv file
emails = pd.read_csv('email_table.csv', index_col='email_id')
emails.head()

emails.describe()

emails.info()

# feature email text has short and long paragraphs. 
# So, let's change it to numerical values. Short t0 2 and long to 4
emails['paragraphs'] = np.where(emails['email_text'] == 'short_email', 2, 4)
del emails['email_text']

# feature email_version has personalize (Hi, John) and generic (Hi)
emails['is_personal'] = (emails['email_version'] == 'personalized').astype(int)
del emails['email_version']

# map weekdays to numbers
weekdaytoindex = {'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday':7}
emails['weekday'] = emails['weekday'].map(weekdaytoindex)

# rename some of the columns to easier names
emails.rename(columns={'user_past_purchases':'purchases', 'user_country':'country'}, inplace=True)

emails.head()

# combine all the information together
emails['response'] = 'received'

open_users = pd.read_csv('email_opened_table.csv').email_id
emails.loc[open_users, 'response'] = 'opened'

clicked_users = pd.read_csv('link_clicked_table.csv').email_id
emails.loc[clicked_users, 'response'] = 'clicked'

emails.head()

emails.to_csv('clean_emails.csv', index_label='email_id')

rslt = emails['response'].value_counts(normalize=True)
rslt

# what percentage opened the emails?
print "{: .2f}% users opened the email".format((1 - rslt['received']) * 100)

# what percentage clicked the link?
print "{: .2f}% users clicked the link in the email".format((rslt['clicked']) * 100) 

# Todo: exploratory data analysis

X = emails.copy()
# ctr_lbl_encoder = LabelEncoder()
# X['country'] = ctr_lbl_encoder.fit_transform(X['response'])

X = X.loc[:,['country', 'purchases', 'paragraphs', 'is_personal']]
X['is_weekend'] = (emails['weekday'] >= 5).astype(int)
X = pd.get_dummies(X, columns=['country'], drop_first=True)
X.head()
# the targer
y = (emails['response'] == 'clicked').astype(int)

# split the data into training and testing
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.33333,random_state = seed)
ytrain.shape

# train the model
# just gonna train one gbm model
train_matrix = xgb.DMatrix(Xtrain, ytrain)
test_matrix = xgb.DMatrix(Xtest)

params = {}
params['objective'] = 'binary:logistic' # output probabilities
params['eval_metric'] = 'auc'
params['num_rounds'] = 300
params['early_stopping_rounds'] = 30
params['max_depth'] = 6
params['eta'] = 0.1
params['subsample'] = 0.8
params['colsample_bytree'] = 0.8

cv_results = xgb.cv(params, train_matrix,
                   num_boost_round=params['num_rounds'],
                   nfold = params.get('nfold', 5),
                   metrics = params['eval_metric'],
                   early_stopping_rounds=params['early_stopping_rounds'],
                   verbose_eval=True,
                   seed=seed)

n_best_trees = cv_results.shape[0]
n_best_trees

watchlist = [(train_matrix, 'train')]

gbt = xgb.train(params, train_matrix, n_best_trees, watchlist)

xgb.plot_importance(gbt)



