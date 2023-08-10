import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import xgboost as xgb

get_ipython().run_line_magic('matplotlib', 'inline')

def read_dataframe():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv') 
    
    nrow_train = train.shape[0]
    combined = pd.concat([train, test])
    
    combined.reset_index(drop=True, inplace=True)
    
    return train, test, combined, nrow_train

train, test, combined, nrow_train = read_dataframe()

combined.head()

combined.describe(include='all')

plt.figure(figsize=(20,10))
sns.heatmap(combined.corr(), annot=True)

np.sum(combined.isnull())/combined.shape[0]

# Relationship between City Category and City Code
print np.sum(combined.isnull()).ix['City_Category'] == combined[combined.City_Category.isnull() & combined.City_Code.isnull()].shape[0]

# Relationship between Employer Category1 and Employer Code
print np.sum(combined.isnull()).ix['Employer_Code'] == combined[combined.Employer_Category1.isnull() & combined.Employer_Code.isnull()].shape[0]

# Relationship between Loan Amount and Loan_Period
print np.sum(combined.isnull()).ix['Loan_Amount'] == combined[combined.Loan_Amount.isnull() & combined.Loan_Period.isnull()].shape[0]

# Relationship between Interest Rate and EMI
print np.sum(combined.isnull()).ix['EMI'] == combined[combined.EMI.isnull() & combined.Interest_Rate.isnull()].shape[0]

# Relationship between Customer Existing Primary Bank Code and Primary Bank Type
print np.sum(combined.isnull()).ix['Primary_Bank_Type'] == combined[combined.Customer_Existing_Primary_Bank_Code.isnull() & combined.Primary_Bank_Type.isnull()].shape[0]

def add_binary_features(combined):
    
    combined['Missing_Loan_EMI'] = combined.EMI.isnull()
    combined['Missing_Loan_Amount'] = combined.Loan_Amount.isnull()
    
    return combined

combined = add_binary_features(combined)

del combined['EMI']
del combined['Interest_Rate']
del combined['Loan_Amount']
del combined['Loan_Period']

def impute_missing_data(combined):
    # Obtain categorical, numerical and date feature names
    categorical_features = ['City_Category', 'City_Code', 'Contacted', 'Customer_Existing_Primary_Bank_Code',  
                            'Employer_Category1', 'Employer_Category2', 'Employer_Code', 'Gender', 'ID', 
                            'Primary_Bank_Type', 'Source', 'Source_Category', 'Var1']
    
    date_features = ['DOB', 'Lead_Creation_Date']
    
    numerical_features = [feat for feat in combined.columns.tolist() 
                          if feat not in categorical_features and feat not in date_features
                          and not feat.startswith('Missing') and feat != 'Approved']
    
    # Imputing most common values for date and categorical features
    for feat in categorical_features:
        combined[feat].fillna('-1', inplace=True)
    
    for feat in date_features:
        mode = combined[feat].value_counts().index[0]
        combined[feat].fillna(mode, inplace=True)
        
    # Imputing median for numerical features
    for feat in numerical_features:
        mean = combined.describe().ix['mean'].ix[feat]
        combined[feat].fillna(mean, inplace=True)
        
    return combined

combined = impute_missing_data(combined)

from sklearn.preprocessing import LabelEncoder

def label_encoding(combined):
    categorical_features = ['City_Category', 'City_Code', 'Contacted', 'Customer_Existing_Primary_Bank_Code',  
                            'Employer_Category1', 'Employer_Category2', 'Employer_Code', 'Gender', 'ID', 
                            'Primary_Bank_Type', 'Source', 'Source_Category', 'Var1', 'Missing_Loan_EMI', 
                            'Missing_Loan_Amount']
    for feat in categorical_features:
        enc = LabelEncoder()
        combined[feat] = enc.fit_transform(combined[feat])
    
    return combined

combined = label_encoding(combined)

plt.figure(figsize=(20,16))
sns.heatmap(combined.corr(),annot=True)

def date_to_datetime(combined):
    combined['DOB'] = combined.DOB.apply(lambda date: date[:-2] + '19' + date[-2:])
    combined['DOB'] = pd.to_datetime(combined.DOB, format='%d/%m/%Y')

    combined['Lead_Creation_Date'] = pd.to_datetime(combined.Lead_Creation_Date, format='%d/%m/%y')
    
    return combined

combined = date_to_datetime(combined)

def get_age(combined):
    
    reference_date = pd.to_datetime(max(combined.Lead_Creation_Date))
    combined['Age'] = combined['Lead_Creation_Date'].subtract(combined['DOB']).apply(lambda timedelta: float(timedelta.days)/365)
    combined['Lead_Creation_Years'] = combined['Lead_Creation_Date'].apply(lambda date: float((reference_date - date).days)/365)
    
    del combined['DOB']
    del combined['Lead_Creation_Date']
    return combined

combined = get_age(combined)

train.describe()

index_to_remove = combined[((combined.Monthly_Income>600000) | (combined.Existing_EMI>52500)) & (combined.Approved.notnull())].index.tolist()
index_to_keep = [index for index in combined.index.tolist() if index not in index_to_remove]

combined = combined.ix[index_to_keep, :]

del combined['ID']

combined['Take_Home_Pay'] = combined['Monthly_Income'] - combined['Existing_EMI']

combined['Monthly_Income'] = np.log1p(combined['Monthly_Income'])
combined['Existing_EMI'] = np.log1p(combined['Existing_EMI'])

plt.figure(figsize=(20, 10))

plt.subplot(221)
sns.distplot(combined[combined.Approved==1].Age, color='blue',
             bins=np.linspace(10, 91, 80))
sns.distplot(combined[combined.Approved==0].Age, color='red',
             bins=np.linspace(10, 91, 80))

plt.subplot(222)
sns.distplot(combined[combined.Approved==1].Monthly_Income, color='blue',
             bins=np.linspace(0, 18, 180))
sns.distplot(combined[combined.Approved==0].Monthly_Income, color='red',
             bins=np.linspace(0, 18, 180))

plt.subplot(223)
sns.distplot(combined[combined.Approved==1].Lead_Creation_Years, color='blue',
             bins=np.linspace(0, 0.5, 183))
sns.distplot(combined[combined.Approved==0].Lead_Creation_Years, color='red',
             bins=np.linspace(0, 0.5, 183))

plt.subplot(224)
sns.distplot(np.log1p(combined[(combined.Approved==1) & (combined.Take_Home_Pay>0)].Take_Home_Pay), color='blue')
sns.distplot(np.log1p(combined[(combined.Approved==0) & (combined.Take_Home_Pay>0)].Take_Home_Pay), color='red')

def train_test_split(combined):
    target_label = 'Approved'
    features = [feat for feat in combined.columns.tolist() if feat != target_label]
    
    train_idx = combined[combined.Approved.notnull()].index.tolist()
    test_idx = combined[combined.Approved.isnull()].index.tolist()
    
    X_train = combined.ix[train_idx, features].values
    X_test = combined.ix[test_idx, features].values
    y_train = combined[target_label].ix[train_idx].values
    
    return X_train, X_test, y_train

X_train, X_test, y_train = train_test_split(combined)

random_state = 1212

params = {
    'objective': 'binary:logistic',
    'min_child_weight': 10.0,
    'max_depth': 7,
    'colsample_bytree': 0.5,
    'subsample': 0.9,
    'eta': 0.02,
    'max_delta_step': 1.2,
    'eval_metric': 'auc',
    'seed': random_state
}

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

submission = pd.DataFrame()
submission['ID'] = test['ID'].values
submission['Approved'] = 0

nrounds= 2000
folds = 10
skf = StratifiedKFold(n_splits=folds, random_state=random_state)

for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
    print 'XGB KFold: %d: ' % int(i+1)
    
    X_subtrain, X_subtest = X_train[train_index], X_train[test_index]
    y_subtrain, y_subtest = y_train[train_index], y_train[test_index]
    
    d_subtrain = xgb.DMatrix(X_subtrain, y_subtrain) 
    d_subtest = xgb.DMatrix(X_subtest, y_subtest) 
    d_test = xgb.DMatrix(X_test)
    
    watchlist = [(d_subtrain, 'subtrain'), (d_subtest, 'subtest')]
    
    mdl = xgb.train(params, d_subtrain, nrounds, watchlist, early_stopping_rounds=150, maximize=True, verbose_eval=50)
    
    # Predict test set based on the best_ntree_limit
    p_test = mdl.predict(d_test, ntree_limit=mdl.best_ntree_limit)
    
    # Take the average of the prediction folds to predict for the test set
    submission['Approved'] += p_test/folds

submission.to_csv('data/submission.csv', index=False)

