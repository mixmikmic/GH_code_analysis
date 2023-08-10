from IPython.display import HTML

HTML('''<script>
code_show=true;
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
}
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')

get_ipython().run_cell_magic('html', '', '\n<a href="https://jy2014.github.io/EpilepsyPrediction/Home.html" target="_self">Back to the Home Page</a>')

#### USING NSCH 2007 Data and NSCH 2007 Variable Description PDF, explore data missingness
from IPython.display import Image

import numpy as np
import pandas as pd
import sklearn.preprocessing as Preprocessing
from sklearn.preprocessing import StandardScaler as Standardize

from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.linear_model import LogisticRegression as Log
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split

import itertools as it
import matplotlib as plt
get_ipython().magic('matplotlib inline')

import warnings
warnings.filterwarnings('ignore')

# Be able to print more text
pd.options.display.max_colwidth = 1000  

Image('pic/nsch_imputation.jpg')

# Load NSCH 2007
nsch_df = pd.read_csv ('NSCH_2007.csv')

# WARNING: sys:1: DtypeWarning: Columns (2,3,4,9,232,233,235) have mixed types. Specify dtype option on import or set low_memory=False.
col_names = nsch_df.columns

# Read CSV containing variable names and descriptions
var_description = pd.read_csv ('variable_description.csv')

print 'Total Shape of NSCH 2007 Dataset', nsch_df.shape
print 'Example Entry in NSCH 2007 Dataset:', '\n', nsch_df.head (n=3)

# Indicator characters denoting: not in universe, legit skip, partial completion, missing in error, added question, deleted question
indicator_char = ['N', 'L', 'P', 'M', 'A', 'D']
                          
# FUNCTION: get_indicator_chars: takes dataframe and counts the number of categorical indicators in each column,
# identifies whether columns are categorical/quantitative.
# INPUTS:
    # df: Dataframe
    # indicator_char: list of indicator characters in dataset (default ['N', 'L', 'P', 'M', 'A', 'D'] )
    # Threshold: int, max number of unique values for variable to be coded as "categorical" (default 20)
# OUTPUTS:
    # List of categorical column indices
    # List of categorical column names
    # df of variables with counts of indicator chars
    
def get_indicator_chars (df, indicator_char = ['N', 'L', 'P', 'M', 'A', 'D'], threshold = 20):
    col_names = df.columns
    
    num_char = np.zeros((len(col_names), len(indicator_char)))
    categorical_col = np.zeros((len(col_names)))

    # Explore missingness: Look for character-coded values
    for i in range(len(col_names)):
        
        # Get counts of unique values in each column
        unique_value_counts = df[col_names[i]].value_counts(sort = True, dropna = False)
        
        # Determine if categorical
        categorical_col[i] = len(unique_value_counts) < threshold
        
        for c in range(len(indicator_char)):
            try:
                # Get number of entries with indicator char (IF EXISTS, ELSE returns error)
                num_char[i,c] = unique_value_counts[indicator_char[c]]
            except:
                # If error returned, set to zero (not found)
                num_char[i,c] = 0
    # Categorical Column Names
    cat_col_names = col_names[np.where(categorical_col)[0]]
    
    # Create Dataframe containing Indicator Chars per Column
    coded_indicators_df = pd.DataFrame(data = num_char, index = col_names, columns = indicator_char, dtype = int)
    
    return categorical_col, cat_col_names, coded_indicators_df

categorical_col, cat_col_names, coded_indicators_df = get_indicator_chars (nsch_df)

print 'Categorical Indicator Counts in each Feature:', '\n', coded_indicators_df

# Examine Missing
sorted_M = coded_indicators_df['M'].sort_values(ascending = False)
# print 'Variables by number of Missing Values:', '\n', sorted_M

ax_missing = sorted_M[:20].plot(kind = 'bar', title = 'Total Responses Missing, Ranked')
ax_missing.set_xlabel('Feature')
ax_missing.set_ylabel('Counts')

# # Get Descriptions of top 10 missing vars
# print 'Descriptions of top 10 missing vars:', '\n'
# for i in range(10):
#     print var_description.loc[var_description['variable'] == sorted_M.index[i]]['descriptions'], '\n'

### HANDLING THE MISSING DATA

# Drop the columns missing over 50,000 values
# After that, the missing vars are related to demographics (poverty, race, marriage, language)
nsch_drop_df = nsch_df.drop(sorted_M[sorted_M > 50000].index, axis = 1)
len(nsch_drop_df.columns)

# Drop columns with A or D
nsch_drop_ad_df = nsch_drop_df.drop('K2Q30D', axis= 1)

# Drop rows with P (did not complete survey)
# How many rows have P = 1066
rows_with_P = np.where(np.sum(nsch_drop_ad_df.values == 'P', axis = 1))[0]
nsch_drop_adp_df = nsch_drop_ad_df.drop(rows_with_P, axis = 0)

# Check how many p left ( should be 0)
print 'Number of P entries left:', np.sum(nsch_drop_adp_df.values == 'P')
print 'New shape:', nsch_drop_adp_df.shape
final_cols = nsch_drop_adp_df.columns


### Explore which of L are categorical
# L - another category
((coded_indicators_df['L'] > 0).values)*(1-categorical_col)

# Column
col_names[np.where(((coded_indicators_df['L'] > 0).values)*(1-categorical_col))[0]]
          
# Columns with L
cat_col_L = col_names[np.where(categorical_col == 1)[0]] # Categorical columns names

noncat_col_L = col_names[np.where(((coded_indicators_df['L'] > 0).values)*(1-categorical_col))[0]]

# Noncategorical L questions             
noncat_L_questions = [var_description.loc[var_description['variable'] == noncat_col_L[i]].descriptions for i in range(len(noncat_col_L))]
 
# for j in range(len(noncat_col_L)):
#     print noncat_L_questions[j] + '\n'

### N 
N_cols = col_names[np.where((coded_indicators_df['N'] > 0 ))[0]]

# Find noncategorical N columns
noncat_col_N = col_names[np.where(((coded_indicators_df['N'] > 0).values)*(1-categorical_col))[0]]         
noncat_N_questions =[var_description.loc[var_description['variable'] == noncat_col_N[i]].descriptions for i in range(len(noncat_col_N))]

# for j in range(len(noncat_col_N)):
#     print noncat_N_questions[j] + '\n'
# Lots of intersection between L, N, so treat the same! Need to manually go thru and see what to do with each

######################
#### DOING IMPUTATION
######################

# Double check that all ADP dropped
drop_vals = ['A', 'D', 'P']
print 'Number of A, D, P:', [np.sum(nsch_drop_adp_df.values == drop_vals[i]) for i in range(len(drop_vals))]

## Create new matrix: drop ADP, coded LMN 
nsch_drop_adp_code_lmn_df = nsch_drop_adp_df

### Categorical Columns
cat_col_names_final= list(set(final_cols).intersection(cat_col_names))

## Categorical: Replace L,N values with -2
nsch_drop_adp_code_lmn_df[cat_col_names_final] = nsch_drop_adp_code_lmn_df[cat_col_names_final].replace(to_replace = ['L', 'N'], value = -2)

## Categorical: Impute M with majority class
# EXCEPT: for Poverty Level -> remove variable from list
cat_col_impute = cat_col_names_final [:] # Copy list
cat_col_impute.remove ('POVERTY_LEVELR') # Remove PovertyLevel from list of cols to impute
print len (cat_col_impute) # should be 271

# convert M to NaN in imputing columns
nsch_drop_adp_code_lmn_df[cat_col_impute] = nsch_drop_adp_code_lmn_df[cat_col_impute].replace('M', np.nan)

# Run preprocessing Imputer with Most Frequent Imputation
imp_cat = Preprocessing.Imputer (missing_values ='NaN', strategy='most_frequent', axis=0)
# Replace columns with imputed columns
nsch_drop_adp_code_lmn_df[cat_col_impute] = imp_cat.fit_transform(nsch_drop_adp_code_lmn_df[cat_col_impute])

categorical_col_final_det, cat_col_names_final_dt, coded_indicators_df_cat = get_indicator_chars (nsch_drop_adp_code_lmn_df[cat_col_names_final])

print coded_indicators_df_cat
len(coded_indicators_df_cat.index) # should be 272
print 'Remaining M in POVERTY_LEVELR:', coded_indicators_df_cat.loc['POVERTY_LEVELR']['M']


### Quantitative Columns
quant_col_names_final = list(set(final_cols).difference(cat_col_names_final))

## EXCEPT: for K9Q16R [Mother's Age] L -> convert these to M
# Convert L in column K9Q16R to M
nsch_drop_adp_code_lmn_df['K9Q16R'] = nsch_drop_adp_code_lmn_df['K9Q16R'].replace('L', 'M')

## Quantitative: replace L, N values with 0
nsch_drop_adp_code_lmn_df[quant_col_names_final] = nsch_drop_adp_code_lmn_df[quant_col_names_final].replace(to_replace = ['L', 'N'], value = 0)

## Quantitative: Impute M with mean
# convert M to NaN
nsch_drop_adp_code_lmn_df[quant_col_names_final] = nsch_drop_adp_code_lmn_df[quant_col_names_final].replace('M', np.nan)
imp_quant = Preprocessing.Imputer (missing_values ='NaN', strategy ='mean', axis=0)
nsch_drop_adp_code_lmn_df[quant_col_names_final] = imp_cat.fit_transform(nsch_drop_adp_code_lmn_df[quant_col_names_final])


categorical_col_final_det, cat_col_names_final_dt, coded_indicators_df_final = get_indicator_chars (nsch_drop_adp_code_lmn_df)

print 'Coded Indicators All Columns:', coded_indicators_df_final
print 'Num Remaining M All Columns:', np.sum(coded_indicators_df_final['M'])
 

### Write Dataframes to CSV format

print 'Final Shape of Dataset:', nsch_drop_adp_code_lmn_df.shape

# nsch_drop_adp_code_lmn_df.to_csv ('NSCH_2007_droppedADP_codeLMN.csv')

# Might be easier to handle if we code M as NaN

nsch_drop_adp_code_lmn_df_na = nsch_drop_adp_code_lmn_df.replace ('M', np.nan)
# nsch_drop_adp_code_lmn_df_na.to_csv ('NSCH_2007_droppedADP_codeLMN_na.csv')
# print nsch_drop_adp_code_lmn_df_na.shape
# print nsch_drop_adp_code_lmn_df_na ['POVERTY_LEVELR']

colfin = nsch_drop_adp_code_lmn_df.columns
cat_col_names_final_dt
[np.sum(colfin == cat_col_names_final_dt[i]) for i in range(len(cat_col_names_final))]
[np.sum(colfin == quant_col_names_final[i]) for i in range(len(quant_col_names_final))]

set(cat_col_names_final_dt).difference(cat_col_names_final)

# Add 'STATE' to Categorical column names (was encoded as Quantitative)
cat_col_names_final.append ('STATE')

categorical_column_names_ser = pd.Series (cat_col_names_final)
# categorical_column_names_ser.to_csv ('Categorical_Column_Names_wState.csv')
 

# load the data set
DF = pd.read_csv('NSCH_2007_droppedADP_codeLMN_na.csv')
print DF.shape
DF.head()
# remove ID number and epilepsy related columns
DF_for_impute = DF.drop(['Unnamed: 0','IDNUMR', 'K2Q42A', 'K2Q42B', 'K2Q42C'], axis = 1);

## Encode categorical variables

# read in categorical column names
df_categorical_names = pd.read_csv('Categorical_Column_Names_wState.csv', header = None)

print df_categorical_names.shape
df_categorical_names.head()

# the categorical columns in the dataset
categorical_names = df_categorical_names.values[:,1]

# remove epilepsy related names
categorical_names = categorical_names[categorical_names != 'K2Q42A']
categorical_names = categorical_names[categorical_names != 'K2Q42B']
categorical_names = categorical_names[categorical_names != 'K2Q42C']
# remove poverty level as well since it is our response variable
categorical_names = categorical_names[categorical_names != 'POVERTY_LEVELR']

# Apply one hot endcoing
DF_for_impute_dummies = pd.get_dummies(DF_for_impute, columns = categorical_names)

print "Dimension of the dataset: ", DF_for_impute_dummies.shape
DF_for_impute_dummies.head()

## Split training, testing and validation sets
# extract poverty columns
poverty = DF_for_impute['POVERTY_LEVELR'].values
# rows with missing values
miss_index =np.isnan(poverty)

print "Number of missing values:", np.sum(miss_index)
## extract numpy arrays from the data
# response is needed only for the complete set
y_complete = DF_for_impute['POVERTY_LEVELR'].values[miss_index == False]

# predictors
DF_for_impute_dummies = DF_for_impute_dummies.drop(['POVERTY_LEVELR'], axis = 1);
x = DF_for_impute_dummies.values

x_complete = x[miss_index == False]
x_missing = x[miss_index == True]

print "Dimenstion of x in the complete set: ", x_complete.shape
print "Dimenstion of x in the set missing poverty level: ", x_missing.shape

## split the complete set into training, validation and testing sets
# train the models on training set
# use validation set to select parameters
# use the accuracy on the test set to compare the final models

n = x_complete.shape[0]
perm = range(n)
np.random.shuffle(perm)

split_ratio_1 = 0.70 # use 75% as training set
split_ratio_2 = 0.15 # use 15% as validation set
split_num_1 = int(split_ratio_1 * n)
split_num_2 = int((split_ratio_1 + split_ratio_2) * n)

x_train = x_complete[perm[: split_num_1], :]
x_validation = x_complete[perm[split_num_1: split_num_2], :]
x_test = x_complete[perm[split_num_2: ], :]

y_train = y_complete[perm[: split_num_1]]
y_validation = y_complete[perm[split_num_1: split_num_2]]
y_test = y_complete[perm[split_num_2: ]]

print "Dimension of predictors in training set: ", x_train.shape
print "Dimension of predictors in validation set: ", x_validation.shape
print "Dimension of predictors in testing set: ", x_test.shape

# randomly assign each observation to classes
y_baseline = np.zeros(len(y_test))
classes = (np.unique(y_complete))

for i in range(len(y_test)):
    # randomly choose from the list
    y_baseline[i] = np.random.choice(classes)
    
# accuracy rate
# print 'Accuracy rate of baseline model is:', np.mean(y_baseline == y_test)

import sys

# parameters to tune random forest
max_depth_list = range(1, 21)
min_leaf_list = range(1, 6)

# prepare to display the progress
total_iter = len(max_depth_list) * len(min_leaf_list)
bar_length = 50
i = 0

# tune the tree
best_score = 0
best_depth = 0
best_leaf_num = 0

# evaluate using cross validation
for depth in max_depth_list:
    for leaf in min_leaf_list:
        # fit the model
        rf = RandomForest(n_estimators=100, max_depth= depth, min_samples_leaf= leaf)
        rf.fit(x_train, y_train)
        # accuracy
        score = rf.score(x_validation, y_validation)

        # update the best score and parameters
        if score > best_score:
            best_score = score
            best_depth = depth
            best_leaf_num = leaf
            rf_best = rf
        
        # display the progress
        percent = float(i) / total_iter
        hashes = '#' * int(round(percent * bar_length))
        spaces = ' ' * (bar_length - len(hashes))
        sys.stdout.write("\rProgress: [{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
        sys.stdout.flush()
        i = i + 1
        
print "\nThe optimal depth is {}; the optimal minimum number of samples in each node is {}.".format(
    best_depth, best_leaf_num)  

print "The best testing accuracy is ", best_score

# # the best model (DETERMINED BY CROSS VALIDATION: TAKES A LONG TIME TO RUN SO WE JUST INPUT THE PARAMETERS HERE!!!)
best_depth = 18
best_leaf_num = 2
rf_best = RandomForest(n_estimators=100, max_depth = best_depth, min_samples_leaf = best_leaf_num)
rf_best.fit(x_train, y_train)

score = rf_best.score(x_test, y_test)
# print "The testing accuracy of the final random forest model is ", score

max_pow_of_10 = 7
min_pow_of_10 = -7
num_params = max_pow_of_10 - min_pow_of_10 + 1

cv_r_squared = []

# Iterate over various parameter values
for power_of_10 in range(min_pow_of_10, max_pow_of_10+1):
        #standardize x_train and y_train
    from time import gmtime, strftime
    print strftime("%Y-%m-%d %H:%M:%S", gmtime())
    
    std = Standardize(with_mean=False)
    x_train_std = std.fit_transform(df_x)
    x_test_std = test_x / std.scale_ 
    #print x_train_std
    cv_r_squared.append(k_fold_r_squared(x_train_std, df_y_training, 5, 10**power_of_10))

# Try logistic regression, with multi_class = 'ovr', first, with regularization parameters(-7 to 7)
def k_fold_r_squared(x_train, y_train, num_folds, param_val):
    n_train = x_train.shape[0]
    n = int(np.round(n_train * 1. / num_folds)) # points per fold
    

    # Iterate over folds
    cv_r_squared = 0
    
    for fold in range(1, num_folds + 1):
        # Take k-1 folds for training 
        x_first_half = x_train[:n * (fold - 1), :]
        x_second_half = x_train[n * fold + 1:, :]
        x_train_cv = np.concatenate((x_first_half, x_second_half), axis=0)
        
        y_first_half = y_train[:n * (fold - 1)]
        y_second_half = y_train[n * fold + 1:]
        y_train_cv = np.concatenate((y_first_half, y_second_half), axis=0)
        
        # Take the middle fold for testing
        x_test_cv = x_train[1 + n * (fold - 1):n * fold, :]
        y_test_cv = y_train[1 + n * (fold - 1):n * fold]

        # Fit ridge regression model with parameter value on CV train set, and evaluate CV test performance
        reg = Log(penalty = 'l2', C = param_val)
        reg.fit(x_train_cv, y_train_cv)
        r_squared = reg.score(x_test_cv, y_test_cv)
    
        # Cummulative R^2 value across folds
        cv_r_squared += r_squared

    # Return average R^2 value across folds
    return cv_r_squared * 1.0 / num_folds

print 'C = 10^-3 returns the best accuracy of weighted logistic regression: 0.648'

def k_fold_r_squared_LDA(x_train, y_train, num_folds):
    n_train = x_train.shape[0]
    n = int(np.round(n_train * 1. / num_folds)) # points per fold
    

    # Iterate over folds
    cv_r_squared = 0
    
    for fold in range(1, num_folds + 1):
        # Take k-1 folds for training 
        x_first_half = x_train[:n * (fold - 1), :]
        x_second_half = x_train[n * fold + 1:, :]
        x_train_cv = np.concatenate((x_first_half, x_second_half), axis=0)
        
        y_first_half = y_train[:n * (fold - 1)]
        y_second_half = y_train[n * fold + 1:]
        y_train_cv = np.concatenate((y_first_half, y_second_half), axis=0)
        
        # Take the middle fold for testing
        x_test_cv = x_train[1 + n * (fold - 1):n * fold, :]
        y_test_cv = y_train[1 + n * (fold - 1):n * fold]

        # Fit ridge regression model with parameter value on CV train set, and evaluate CV test performance
        reg = LDA()
        reg.fit(x_train_cv, y_train_cv)
        r_squared = reg.score(x_test_cv, y_test_cv)
    
        # Cummulative R^2 value across folds
        cv_r_squared += r_squared

    # Return average R^2 value across folds
    return cv_r_squared * 1.0 / num_folds

df_x_array = np.array(df_x)
cv_r_squared_lda = k_fold_r_squared_LDA(df_x_array, df_y_training, 5)
print 'The testing accuracy of the final LDA model is:', cv_r_squared_lda

def k_fold_r_squared_QDA(x_train, y_train, num_folds):
    n_train = x_train.shape[0]
    n = int(np.round(n_train * 1. / num_folds)) # points per fold
    

    # Iterate over folds
    cv_r_squared = 0
    
    for fold in range(1, num_folds + 1):
        # Take k-1 folds for training 
        x_first_half = x_train[:n * (fold - 1), :]
        x_second_half = x_train[n * fold + 1:, :]
        x_train_cv = np.concatenate((x_first_half, x_second_half), axis=0)
        
        y_first_half = y_train[:n * (fold - 1)]
        y_second_half = y_train[n * fold + 1:]
        y_train_cv = np.concatenate((y_first_half, y_second_half), axis=0)
        
        # Take the middle fold for testing
        x_test_cv = x_train[1 + n * (fold - 1):n * fold, :]
        y_test_cv = y_train[1 + n * (fold - 1):n * fold]

        # Fit ridge regression model with parameter value on CV train set, and evaluate CV test performance
        reg = QDA()
        reg.fit(x_train_cv, y_train_cv)
        r_squared = reg.score(x_test_cv, y_test_cv)
    
        # Cummulative R^2 value across folds
        cv_r_squared += r_squared

    # Return average R^2 value across folds
    return cv_r_squared * 1.0 / num_folds

# Try QDA
cv_r_squared_qda = k_fold_r_squared_QDA(df_x_array, df_y_training, 5)
print 'The testing accuracy of the final QDA model is:', cv_r_squared_qda

get_ipython().run_cell_magic('html', '', '<a href="https://jy2014.github.io/EpilepsyPrediction/data_source.html" target="_self">Chapter 1. Data Source</a>')

get_ipython().run_cell_magic('html', '', '<a href="https://jy2014.github.io/EpilepsyPrediction/Diagnosis.html" target="_self">Chapter 3. Predicting Epilepsy Status</a>')

get_ipython().run_cell_magic('html', '', '<a href="https://jy2014.github.io/EpilepsyPrediction/Home.html" target="_self">Back to the Home Page</a>')

# Random forest
rf_best.fit(x_complete, y_complete)
y_pred =rf_best.predict(x_missing)

DF['POVERTY_LEVELR'].iloc[miss_index] = y_pred

# DF.to_csv("imputed_PovertyLevel_RF.csv", sep = ',')

