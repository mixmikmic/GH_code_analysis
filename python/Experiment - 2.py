import numpy as np
import pandas as pd
import os, sys

from sklearn.cross_validation import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

basepath = os.path.expanduser('~/Desktop/src/AllState_Claims_Severity/')
sys.path.append(os.path.join(basepath, 'src'))

np.random.seed(2016)

from data import *
from utils import *

# load files
train      = pd.read_csv(os.path.join(basepath, 'data/raw/train.csv'))
test       = pd.read_csv(os.path.join(basepath, 'data/raw/test.csv'))
sample_sub = pd.read_csv(os.path.join(basepath, 'data/raw/sample_submission.csv'))

# create an indicator for somewhat precarious values for loss. ( only to reduce the number of training examples. )
train['outlier_flag'] = train.loss.map(lambda x: int(x < 4e3))

# encode categorical variables
train, test = encode_categorical_features(train, test)

# get stratified sample
itrain, itest = get_stratified_sample(train.outlier_flag)

# subsample of data to work with
train_sub = train.iloc[itrain]

# target variable
y = np.log(train.loss)

def forward_feature_selection(df):
    columns = df.columns
    
    # rearrange columns in such a way that target variables ( loss, outlier_flag ) is
    # followed by continuous and categorical variables
    
    cont_columns = [col for col in columns if 'cont' in col]
    cat_columns  = [col for col in columns if 'cat' in col]
    
    df = df[list(columns[-2:]) + cont_columns + cat_columns]
    
    y              = np.log(df.loss)
    outlier_flag   = df.outlier_flag
    
    selected_features = []
    features_to_test  = df.columns[2:]
    
    n_fold = 5
    cv     = StratifiedKFold(outlier_flag, n_folds=n_fold, shuffle=True, random_state=23232)
    
    mae_cv_old      = 5000
    is_improving    = True
    
    while is_improving:
        mae_cvs = []
        
        for feature in features_to_test:
            print('{}'.format(selected_features + [feature]))
            
            X = df[selected_features + [feature]]
            
            mae_cv = 0
            
            for i, (i_trn, i_val) in enumerate(cv, start=1):
                est = xgb.XGBRegressor(seed=121212)
                
                est.fit(X.values[i_trn], y.values[i_trn])
                yhat = np.exp(est.predict(X.values[i_val]))

                mae = mean_absolute_error(np.exp(y.values[i_val]), yhat)
                mae_cv += mae / n_fold

            print('MAE CV: {}'.format(mae_cv))
            mae_cvs.append(mae_cv)
        
        mae_cv_new = min(mae_cvs)

        if mae_cv_new < mae_cv_old:
            mae_cv_old = mae_cv_new
            feature = list(features_to_test).pop(mae_cvs.index(mae_cv_new))
            selected_features.append(feature)
            print('selected features: {}'.format(selected_features))
            
            with open(os.path.join(basepath, 'data/processed/features_xgboost/selected_features.txt'), 'w') as f:
                f.write('{}\n'.format('\n'.join(selected_features)))
                f.close()
        else:
            is_improving = False
            print('final selected features: {}'.format(selected_features))
    
    
    print('saving selected feature names as a file')
    with open(os.path.join(basepath, 'data/processed/features_xgboost/selected_features.txt'), 'w') as f:
        f.write('{}\n'.format('\n'.join(selected_features)))
        f.close()

forward_feature_selection(train)

selected_features = [
                        'cat80',
                        'cat101',
                        'cat100',
                        'cat57',
                        'cat114',
                        'cat79',
                        'cat44',
                        'cat26',
                        'cat94',
                        'cat38',
                        'cat32',
                        'cat35',
                        'cat67',
                        'cat59'
                   ]

X = train[selected_features]

itrain, itest = train_test_split(range(len(X)), stratify=train.outlier_flag, test_size=0.2, random_state=11232)

X_train = X.iloc[itrain]
X_test  = X.iloc[itest]

y_train = y.iloc[itrain]
y_test  = y.iloc[itest]

clf = RandomForestRegressor(n_estimators=100, max_depth=13, n_jobs=-1, random_state=12121)
clf.fit(X_train, y_train)

y_hat = np.exp(clf.predict(X_test))
print('MAE on unseen examples ', mean_absolute_error(np.exp(y_test), y_hat))



