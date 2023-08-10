# Import libraries
import os
import sys
import inspect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

# Modify notebook settings
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("whitegrid")
pd.options.display.max_columns = 100

# Create a variable for the project root directory
proj_root = os.path.join(os.pardir)

# Save path to the wrangled data file
# "dataset_wrangled.csv"
wrangled_data_file = os.path.join(proj_root,
                                "data",
                                "interim",
                                "dataset_wrangled.csv")

# add the 'src' directory as one where we can import modules
src_dir = os.path.join(proj_root, "src")
sys.path.append(src_dir)

# Read in the wrangled credit card client default data set.

df_wrangled = pd.read_csv(wrangled_data_file,
                          header=1, 
                          index_col=0)
#df_wrangled.head()

# Set alpha
alpha = 0.05

# Set upper bound and lower bound of percentiles
# for calculating confidence intervals, given alpha.
lb = (alpha/2)
ub = 1 - (alpha/2)
bounds = [lb, ub]

# Create a function to calculate the proportion of defaults by sex.
def make_proportion_dict(df):
    """
    Given a DataFrame, return a dictionary where sex labels
    ('Male' and 'Female') are the keys and the proportion of 
    defaults are the values.
    """
    output_dict = {'Male': df[df['SEX'] == 1]['default payment next month'].value_counts(normalize=True).loc[1],
                   'Female': df[df['SEX'] == 2] ['default payment next month'].value_counts(normalize=True).loc[1]}
    return output_dict

make_proportion_dict(df_wrangled)

def resample(df, iters=1000):
    list_m = []
    list_w = []
    n = len(df)
    for i in range(iters):
        resampled_df = df[['default payment next month', 
                           'SEX']].sample(n=n, replace=True)
        proportions = make_proportion_dict(resampled_df)
        list_m.append(proportions['Male'])
        list_w.append(proportions['Female'])
    prop_array_m = np.array(list_m)
    prop_array_w = np.array(list_w)
    return prop_array_m, prop_array_w 

results_m, results_w = resample(df_wrangled, iters=1000)

sns.distplot(results_m,
             norm_hist=True, 
             label='Male')
sns.distplot(results_w,
             norm_hist=True, 
             label='Female')

plt.xlabel('Proportion of Defaults')
plt.legend()

# Import modules
from rpy2.robjects import FloatVector, Formula
from rpy2.robjects.packages import importr
# import glm from the source code
import models.glm as glm

# Create robjects for X ('AGE'), 
# y ('default payment next month'), 
# regression formula, and the environment
y = df_wrangled['default payment next month']
X = df_wrangled['AGE']

y_r = FloatVector(y)
X_r = FloatVector(X)

fmla = Formula('y_r ~ X_r')
env = fmla.environment

env['y_r'] = y_r
env['X_r'] = X_r

glm_result = glm.logit_model(model='y_r ~ X_r')

p_val = 0.0162

p_val < alpha

# Create robjects for X ('LIMIT_BAL'), 
# y ('default payment next month'), 
# regression formula, and the environment
y = df_wrangled['default payment next month']
X = df_wrangled['LIMIT_BAL']

y_r = FloatVector(y)
X_r = FloatVector(X)

fmla = Formula('y_r ~ X_r')
env = fmla.environment

env['y_r'] = y_r
env['X_r'] = X_r

glm_result = glm.logit_model(model='y_r ~ X_r')

def build_features_ba_over_cl(df):

    bill_amount_column_list = ['BILL_AMT1',
                               'BILL_AMT2',
                               'BILL_AMT3',
                               'BILL_AMT4',
                               'BILL_AMT5',
                               'BILL_AMT6']

    df_new = df.copy()
    
    for i, ba in enumerate(bill_amount_column_list, 1):

        new_column_name = 'ba_over_cl_' + str(i)
        
        df_new[new_column_name] =             df_wrangled[ba] / df_wrangled['LIMIT_BAL']

    return df_new

new_df = build_features_ba_over_cl(df_wrangled)
# new_df.head()

ba_over_cl_list = ['ba_over_cl_1', 
                   'ba_over_cl_2',
                   'ba_over_cl_3', 
                   'ba_over_cl_4', 
                   'ba_over_cl_5', 
                   'ba_over_cl_6']
    
for col in ba_over_cl_list:
    print('\n')
    print('X_r:', col)
    print('\n')
    
    y = new_df['default payment next month']
    X = new_df[col]

    y_r = FloatVector(y)
    X_r = FloatVector(X)

    fmla = Formula('y_r ~ X_r')
    env = fmla.environment

    env['y_r'] = y_r
    env['X_r'] = X_r
    glm_result = glm.logit_model(model='y_r ~ X_r')
    
#    print(glm_result)

def build_features_ba_less_pa_over_cl(df):

    bill_amount_column_list = ['BILL_AMT1',
                               'BILL_AMT2',
                               'BILL_AMT3',
                               'BILL_AMT4',
                               'BILL_AMT5',
                               'BILL_AMT6']

    pay_amount_column_list = ['PAY_AMT1',
                              'PAY_AMT2',
                              'PAY_AMT3',
                              'PAY_AMT4',
                              'PAY_AMT5',
                              'PAY_AMT6']

    df_new = df.copy()
    
    for i, (ba, pa) in enumerate(zip(bill_amount_column_list, 
                               pay_amount_column_list), 
                           1):

        new_column_name = 'ba_less_pa_over_cl_' + str(i)
        
        df_new[new_column_name] =             (df_wrangled[ba] -df_wrangled[pa])/ df_wrangled['LIMIT_BAL']

    return df_new

new_df = build_features_ba_less_pa_over_cl(new_df)
#new_df.head()

ba_less_pa_over_cl_list = ['ba_less_pa_over_cl_1', 
                           'ba_less_pa_over_cl_2',
                           'ba_less_pa_over_cl_3', 
                           'ba_less_pa_over_cl_4', 
                           'ba_less_pa_over_cl_5', 
                           'ba_less_pa_over_cl_6']
    
for col in ba_less_pa_over_cl_list:
    print('\n')
    print('X_r:', col)
    print('\n')
    
    y = new_df['default payment next month']
    X = new_df[col]

    y_r = FloatVector(y)
    X_r = FloatVector(X)

    fmla = Formula('y_r ~ X_r')
    env = fmla.environment

    env['y_r'] = y_r
    env['X_r'] = X_r
    glm_result = glm.logit_model(model='y_r ~ X_r')
    
#    print(glm_result)

# import build_features from the source code
import features.build_ratio_features as brf

# Engineer features
brf.create_interim_dataset(new_file_name='dataset_interim.csv')

# Save path to the interim data file
# "dataset_interim.csv"
interim_data_file = os.path.join(proj_root,
                                   "data",
                                   "interim",
                                   "dataset_interim.csv")

# Read in the new credit card client default data set.
df_new = pd.read_csv(interim_data_file, 
                           index_col=0)

df_new.head()

