# general imports
import pickle
import pandas as pd
import numpy as np

#imports for chi-squared
from scipy.stats import chi2_contingency
from collections import defaultdict

# imports for xgboost
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from xgboost import cv
from sklearn.model_selection import RandomizedSearchCV

# import training data
df_train = pd.read_csv(r'train.csv', low_memory=False)
df_test = pd.read_csv(r'test.csv', low_memory=False)

#helper functions
def common_columns(df1, df2):
    """Returns tuple (df1, df2) with columns that BOTH df1 and df2 have in common"""
    joint_column_list = list(set(df1.columns) & set(df2.columns))
    
    return (df1[joint_column_list], df2[joint_column_list])

def sufficiently_filled(df, threshold):
    """
    Removes columns from df that are below the threshold for being filled

    Paramerters
    -----------
    df : pandas dataframe
        dataframe with NaN values
    threshold : int
        number of exceptable NaN values for each column
    Returns
    -------
    dataframe
        dataframe with columns that have less than the threshold number of NaN values
    """
    # remove all columns with no data
    df1 = df.dropna(axis=1, how='all')
    # counts the number of NaNs in each column and keeps only the ones with less NaNs then threshold
    good_cols = df1.isna().astype('int').sum() < threshold
    cols_to_keep = (good_cols[good_cols == True]).index
    return df[cols_to_keep]

# create dataframe where each column has at least 50% of entries filled
threshold = len(df_train)/2. # define threshold
# removes empty or insufficently filled columns
training_data = sufficiently_filled(df_train, threshold) 
test_data = sufficiently_filled(df_test, threshold)

# return dataframes with columns that are only in both test and training data
clean_train, clean_test = common_columns(training_data, test_data)

num_removed_cols = len(list(df_train)) - len(list(clean_train))
num_removed_rows = len(df_train)-len(clean_train)
print('The cleaning process removed {} columns and {} rows in the training data'.format(num_removed_cols, num_removed_rows))

# Isolate text data
text_train = clean_train.select_dtypes(exclude=['float64','int64'])
text_test = clean_test.select_dtypes(exclude=['float64', 'int64'])
print('There are {} columns of text training data'.format(len(text_train.columns)))

# Create list of catagorical feature names
data_dictionary = pd.read_excel('WiDS data dictionary v2.xlsx')
col_list = list(data_dictionary['Column Name'][1:].apply(lambda x: str(x)))
# Create list of columns in cleaned training data
clean_data_columns = clean_train.columns
# Create list of columns both categorical and in the cleaned training data
categorical_column_names = [name for name in clean_data_columns if name in col_list]
# Cast catagorical data as object datatype
categorical_train = clean_train[categorical_column_names].drop(columns='DG1').astype('object')
categorical_test = clean_test[categorical_column_names].drop(columns='DG1').astype('object')
print('There are {} columns of categorical training data'.format(len(categorical_train.columns)))

# Dataframe of numerical data
drop_columns = categorical_column_names + list(text_train)
numerical_train = clean_train.drop(columns=drop_columns)
numerical_test = clean_test.drop(columns=drop_columns)
print('There are {} columns of numerical training data'.format(len(numerical_train.columns)))

# function to count values for each possible category
def cat_count(pd_series):
    '''Returns all possible values in a pandas series'''
    categories = list(set(pd_series))
    cat_count = dict.fromkeys(categories, 0)
    for cat in pd_series:
        cat_count[cat] += 1
    return cat_count

# function to create joint dist table
def joint_dist_table(cat_series, df):
    '''
    Create a joint distribution table for pandas series
    
    Paramerters
    -----------
    cat_series:
    df:
    
    Returns
    -------
    Joint distribution table of the catagorical distribution for each gender
    
    '''
    data = df.copy()
    # split male and female counts, and drops and Nans
    F_series = cat_series[data.is_female == 1].dropna()
    M_series = cat_series[data.is_female == 0].dropna()
    # create count of each category, for each gender
    F = cat_count(F_series)
    M = cat_count(M_series) 
    keep = set(F) & set(M)
    F_new = {k: F[k] for k in keep}
    M_new = {k: M[k] for k in keep}
    # combine counts in dataframe
    dist_table = pd.DataFrame.from_dict(F_new, orient='index')
    dist_table[1] = M_new.values()
    # format to distribution table
    final_dist_table = dist_table.rename(columns={0:'Male',1:'Female'}).transpose()
    return final_dist_table   
    

# calculate p-value for each categorical value
chi_dict = defaultdict(list)
for cat_cols in categorical_train:
    jd_table = joint_dist_table(categorical_train[cat_cols], df_train)
    chi_test_value, chi_p, degfree, exp_val = chi2_contingency(jd_table)
    chi_dict[cat_cols] = [chi_test_value, chi_p, degfree, exp_val]

# filter columns based on p-value
sig_level = 0.05 # significance level
sig_cols = []

for k,v in chi_dict.items():
    if v[1] < sig_level:
            sig_cols.append(k)
# print the number of significant features
print('There are {} significant categorical features'.format(len(sig_cols)))

# create a dataframe for only the categorical data dependent on gender
significant_categorical_train = df_train[sig_cols].astype('object') 
significant_categorical_test = df_test[sig_cols].astype('object')
# one-hot encode
encoded_categorical_train = pd.get_dummies(significant_categorical_train, dummy_na=True)
encoded_categorical_test = pd.get_dummies(significant_categorical_test, dummy_na=True)
# make sure training and test features are the same
joint_features = list(set(encoded_categorical_test.columns) & set(encoded_categorical_train))
encoded_categorical_train = encoded_categorical_train[joint_features]
encoded_categorical_test = encoded_categorical_test[joint_features]

# one-hot encode categorical data, without feature selection
unfiltered_categorical_train = pd.get_dummies(categorical_train, dummy_na=True)
unfiltered_categorical_test = pd.get_dummies(categorical_test, dummy_na=True)
joint_features = list(set(unfiltered_categorical_test.columns) & set(unfiltered_categorical_train))
unfiltered_categorical_train = unfiltered_categorical_train[joint_features]
unfiltered_categorical_test = unfiltered_categorical_test[joint_features]

# Use un-optomized xgboost model with the numerical, categorical, and a combination of the two
combined_data = pd.concat([encoded_categorical_train, numerical_train], axis=1) 
datasets = [unfiltered_categorical_train, significant_categorical_train.fillna(value=100).astype(int), encoded_categorical_train, numerical_train, combined_data]
# define parameters
fixed_parameters = {
               'max_depth':3,
               'learning_rate':0.3,
               'min_child_weight':3,
               'colsample_bytree':0.8,
               'subsample':0.8,
               'gamma':0,
               'max_delta_step':0,
               'colsample_bylevel':1,
               'scale_pos_weight':1,
               'base_score':0.5,
               'random_state':5,
               'objective':'binary:logistic',
               'silent': 1}

accuracy_scores = []
for data in datasets:
    # define features(X), and target(y)
    X = data
    y = df_train.is_female
    # instantiate model
    xg_reg = XGBRegressor(**fixed_parameters)
    # fit model
    xg_reg.fit(X, y)
    # predict y values
    y_pred = xg_reg.predict(X)
    predictions = [round(value) for value in y_pred]
    # score model
    score = accuracy_score(y, predictions)
    print(score)
   

# dictionary of fixed parameters, which will not be optimized
fixed_parameters = {
    'objective':'binary:logistic',
    'max_delta_step':0,
    'scale_pos_weight':1,
    'base_score':0.5,
    'random_state':5,
    'subsample':0.8,
    'silent': 1
}

# dictionary of parameters to optimize, and the range of optimization values
reg_param_grid = {'max_depth': range(2,10),
                  'learning_rate': [0.05, 0.1, 0.15, 0.3],
                  'min_child_weight':[2,3,4],
                  'colsample_bytree':[0.6, 0.7, 0.8],
                  'gamma':[0, 2, 5, 8],
                  'colsample_bylevel':[0.7, 1]
                 }
    

# define features and labels for training data
X = combined_data
y = df_train.is_female

# instantiate classifier
xg_reg = XGBRegressor(**fixed_parameters)

# RandomSearch
grid_search = RandomizedSearchCV(param_distributions = reg_param_grid, estimator = xg_reg, cv=4, n_iter=200)
grid_search.fit(X,y)

# Print best parameters and results
print(grid_search.best_params_)
print(grid_search.best_score_)

optimized_fixed_parameters = {
    'objective':'binary:logistic',
    'max_delta_step':0,
    'scale_pos_weight':1,
    'base_score':0.5,
    'random_state':5,
    'subsample':0.8,
    'silent': 1,
    'min_child_weight': 3,
    'max_depth': 8,
    'learning_rate': 0.1,
    'gamma': 0,
    'colsample_bytree': 0.6,
    'colsample_bylevel': 1
}

# instantiate classifier
xg_reg = XGBRegressor(**optimized_fixed_parameters)
xg_reg.fit(X,y)

# predict the labels
y_pred = xg_reg.predict(X)
# convert probabilities to binary output
predictions = [round(value) for value in y_pred]
# score model
score = accuracy_score(y, predictions)
# print accuracy
print(score)

# Define test features
X_sub = pd.concat([encoded_categorical_test, numerical_test], axis=1) 

# Predict label of test data with optimized model
test_predictions = xg_reg.predict(X_sub)

# Place predictions and their corresponding test ID in a dataframe
submission_df = pd.DataFrame({'test_id': df_test.test_id, 'is_female': test_predictions})

# export the results to a csv file, so that it can be submitted to Kaggle.com
submission_df.to_csv('sub20.csv')



