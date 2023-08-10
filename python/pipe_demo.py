import copy
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# imports for piplelines
from sklearn.pipeline import Pipeline, FeatureUnion

# built-in transformer which we will use in our pipelines
from sklearn.preprocessing import PolynomialFeatures

#evaluation metrics  
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix, precision_recall_curve

# models and model selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# custom imports
from custom_transformers import DataFrameSelector, ZeroVariance, FindCorrelation
from custom_transformers import OptionalStandardScaler, ManualDropper, PipelineChecker

from ml_plot import kde_plot, hist_plot, cat_plot, pairwise_plot, LiftChart, ROCPlot, train_plot

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
warnings.filterwarnings('ignore')

# load into dataframe
churnTrain = pd.read_csv('churnTrain.csv')
churnTest = pd.read_csv('churnTest.csv')

# split features and outcome, and transform outcome to binary
features = [x for x in churnTrain if x != 'churn']
features_train = churnTrain[features]
features_test = churnTest[features]
outcome_train = churnTrain.loc[:, 'churn'].apply(lambda x: 1 if x == "yes" else 0)
outcome_test = churnTest.loc[:, 'churn'].apply(lambda x: 1 if x == "yes" else 0)

print('{} \n\nunique dtypes: {}'.format(features_train.info(),
                                        set([features_train[x].dtype for x in features_train])))

fac_cols = [x for x in features_train if features_train[x].dtype == np.dtype('O')]
float_cols = [x for x in features_train if features_train[x].dtype == np.dtype('float64')]
int_cols = [x for x in features_train if features_train[x].dtype == np.dtype('int64')]

# convert objects into pandas categorical type
# get a warning (suppressed here), but we are OK
features_train.loc[:, fac_cols] = features_train.loc[:, fac_cols].apply(lambda x: pd.Categorical(x))
features_test.loc[:, fac_cols] = features_test.loc[:, fac_cols].apply(lambda x: pd.Categorical(x))

def category_checker(X_train, X_test, cat_col):
    allowed_values = dict()
    for col in cat_col:
        allowed_values[col] = X_train[col].values.categories
    
    warn = False
    for col in cat_col:
        tmp = X_test[col].values.categories
        for val in tmp.values.tolist():
            if val not in allowed_values[col].values.tolist():
                print('WARNING: new categorical level encountered:', col, val)
                warn = True
    
    if not warn:
        print('No problems detected')

# we see that there are no categories in the test set that do not appear 
# in the training set
category_checker(features_train, features_test, fac_cols)

#-----------------------------------------------------------
#
# As an example, kernel density plots for numerical features.
#
#-----------------------------------------------------------

kde_plot(churnTrain.loc[:, float_cols  + ['churn']],
         outcome_col = 'churn',
         n_col = 3,
         plot_legend = True,
         f_size = (12,12))

#-----------------------------------------------------------
#
# Histograms for numerical features.
# Uncomment code below and run cell to produce.
# You may want to try hist_plot as well
#-----------------------------------------------------------

# hist_plot(churnTrain.loc[:, float_cols  + ['churn']],
#          outcome_col = 'churn',
#          n_col = 3,
#          plot_legend = True,
#          f_size = (12,12))

#-----------------------------------------------------------
#
# Bar charts for categorical features
# Uncomment code below and run cell to produce
#
#-----------------------------------------------------------

# cat_plot(churnTrain.loc[:, fac_cols + ['churn']],
#          outcome_col = 'churn',
#          n_col = 2,
#          plot_legend = True,
#          f_size = (22,12))

#-----------------------------------------------------------
#
# Pairwise plot to investigate correlations
# amongst numberical features (you may want to investigate)
# correlation between categorical features as well!)
# Uncomment code below and run cell to produce
#
#-----------------------------------------------------------

# pairwise_plot(churnTrain.loc[:, float_cols + int_cols + ['churn']],
#          outcome_col = 'churn',
#          f_size = (10,10))

# pandas has helper function for creating dummies- much easier than
# using sklearn helper classes
fac_dummies_train = pd.get_dummies(features_train.loc[:, fac_cols])

# Need to drop: international_plan_no, voice_mail_plan_no as these are binary.
# We could optionally drop: area_code_408, state_AK: for linear models these 
# could be considered as the base class.
# For now, we can store these column names in a list.
drop_cols = ["international_plan_no", 'voice_mail_plan_no']
opt_drop_cols = ['state_AK', 'area_code_area_code_408']

# what are the column indices? Remember, sk-learn uses numpy arrays,
# not pandas dataframes, so we can only index by column index
fac_col_names = fac_dummies_train.columns.values
drop_col_ix = [np.where(fac_col_names == x)[0][0] for x in drop_cols]
opt_drop_col_ix = [np.where(fac_col_names == x)[0][0] for x in opt_drop_cols]

# Combine numerical and encoded categorical dataframes.
# This is where we will pass over to our pipeline
features_train = fac_dummies_train.merge(features_train.loc[:, int_cols + float_cols],
                                         left_index = True, right_index = True)

# data pipeline for numerical features
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(float_cols + int_cols)),
    ('zero_var', ZeroVariance(near_zero=True)),
    ('correlation', FindCorrelation(threshold=0.9)),
    ('opt_scaler', OptionalStandardScaler(scale=False)),
    ('poly_features', PolynomialFeatures(degree=1, include_bias=False)),
])

# data pipeline for categorical features
cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(fac_col_names)),
    ('manual_dropper', ManualDropper(drop_ix = drop_col_ix,
                                     optional_drop_ix=opt_drop_col_ix)),
    ('zero_var', ZeroVariance(near_zero=True)),
    ('correlation', FindCorrelation(threshold=0.9)),
])

#-----------------------------------------------------------
#
# example: run through num_pipeline.
# Get back an numpy array. 
# Note that all same datatype now: numpy can only consider
# homogenous data types, so integer converted to float.
#
#-----------------------------------------------------------

num_arr = num_pipeline.fit_transform(features_train)
# first 5 rows
print(num_arr[:5, :])

#-----------------------------------------------------------
#
# We can access steps in the fitted pipeline using get_params() method.
# We want to access the get_feature_names() method in each step, which 
# will allow us to filter an array of names at each step to correspond to 
# the features that are being selected
#
#-----------------------------------------------------------

# step 1: those filtered by low variance
names_1 = num_pipeline.get_params()['zero_var'].get_feature_names(np.array(float_cols + int_cols))
# step 2: those filtered by correlation. Pass in the remaining names from the previous step
names_2 = num_pipeline.get_params()['correlation'].get_feature_names(names_1)
# step3: those from interaction
names_3 = num_pipeline.get_params()['poly_features'].get_feature_names(names_2)

# and prove we have recovered names...
pd.DataFrame(num_arr, columns=names_3).head()

#-----------------------------------------------------------
#
# Use feature union to make a big pipeline
#
#-----------------------------------------------------------

prep_pipe = Pipeline([
        
    ('union', FeatureUnion(
        transformer_list = [
                    
            # pipeline to transform numeric features
            ('num_pipeline', num_pipeline),
                    
            # pipeline for categorical
            ('cat_pipeline', cat_pipeline),
        ],
                
        #other arguments for FeatureInion        
        n_jobs = 1,
        transformer_weights = None
    )),
        
    # final correlation check
    ('correlation', FindCorrelation(threshold=0.9)),
        
    # error checking
    ('checker', PipelineChecker()),
])

# logistic regression. take a copy of the pipeline, and append an estimator to the end
lr_est = copy.deepcopy(prep_pipe)
lr_est.steps.append(('logistic_regression', LogisticRegression(random_state = 1234)))

# set the hyperparameter grid. Good news: we can treat preprocessing steps as we would
# any other hyperparameter. Be careful though, as can easily blow up number of models 
# will be building (especially with CV as well)
lr_param_grid = dict(union__num_pipeline__opt_scaler__scale=[True, False],
                  union__num_pipeline__poly_features__degree=[1,2],
                  logistic_regression__penalty=['l1'],
                  logistic_regression__C=[0.001, 0.01, 0.1, 1, 10])

# cross validation object
grid_search_lr = GridSearchCV(estimator=lr_est,
                              param_grid=lr_param_grid,
                              scoring='roc_auc',
                              n_jobs=1,
                              cv=5,
                              refit=True,
                              verbose=1)

grid_search_lr.fit(features_train, outcome_train);

# GridSearchCV object has many useful attributes, for example:
print('Chosen params: {}\n\nTrain AUC score: {:0.3f}'.format(grid_search_lr.best_params_,
                                                        grid_search_lr.best_score_))

#-----------------------------------------------------------
#
# Visualise CV results
#
# Cannot look at everything, so we will consider the mean AUC of test folds 
# as a function of polynomial degree and penalty. 
#
#-----------------------------------------------------------

# results are stored as dict, lets pull into a dataframe
lr_results = pd.DataFrame(grid_search_lr.cv_results_)

plt_cols = ['param_union__num_pipeline__poly_features__degree',
            'param_logistic_regression__C',
            'mean_test_score',
            'std_test_score']

# filter and rename for neat plotting
lr_results = (lr_results.query('param_union__num_pipeline__opt_scaler__scale == True')
                 .loc[:, plt_cols]
                 .rename(columns = {'param_logistic_regression__C' : 'C',
                                    'param_union__num_pipeline__poly_features__degree' : 'polynomial degree',
                                    'mean_test_score' : 'AUC'
                                    })
)

# train plot is a plotting helper function for visualising hyperparameter training.
# it can consider up to three hyperparameters, and optionally can plot
# error bars. See ml_plots.py for the source code
train_plot(lr_results, logx=True)

# rebuild features. Notice we selected polynomial degree 2 after cross validation,
# so we have lot of extra features

# can access parts of the CV pipeline object like so:
num_pipe_lr = grid_search_lr.best_estimator_.get_params()['union__num_pipeline']
cat_pipe_lr = grid_search_lr.best_estimator_.get_params()['union__cat_pipeline']

# names from numerical pipeline
names_1a = num_pipe_lr.get_params()['zero_var'].get_feature_names(np.array(float_cols + int_cols))
names_2a = num_pipe_lr.get_params()['correlation'].get_feature_names(names_1a)
names_3a = num_pipe_lr.get_params()['poly_features'].get_feature_names(names_2a)

# names from categorical pipeline
names_1b = cat_pipe_lr.get_params()['manual_dropper'].get_feature_names(np.array(fac_col_names))
names_2b = cat_pipe_lr.get_params()['zero_var'].get_feature_names(np.array(names_1b))
names_3b = cat_pipe_lr.get_params()['correlation'].get_feature_names(np.array(names_2b))

# names following final correlation check
names_1c = np.array(names_3a + names_3b.tolist())
lr_feature_names = grid_search_lr.best_estimator_.get_params()['correlation'].get_feature_names(names_1c)

#-----------------------------------------------------------
#
# We can use this usefully: lets examine the coefficient of each 
# feature
#
#-----------------------------------------------------------

# coefficients are in the best_estimator_ attribute
lr_coef = grid_search_lr.best_estimator_.get_params()['logistic_regression'].coef_

# we can sort the list by coef, absolute coef etc. Display the top 10
sorted(list(zip( lr_coef[0], lr_feature_names )), key = lambda x: x[0], reverse=True)[:10]
#sorted(list(zip( lr_coef[0], lr_feature_names )), key = lambda x: abs(x[0], reversed=True))

# use same preperation pipeline, just a different estimator
rf_est = copy.deepcopy(prep_pipe)
rf_est.steps.append(('random_forest', RandomForestClassifier(random_state = 1234)))

# parameters
# We can set the params to a single value for those hyperparameters we want to fix,
# for example we want to turn off the scaler, dont drop any optional columns,
# and fix poly degree to 1 (i.e. no poly terms)
rf_param_grid = dict(union__num_pipeline__opt_scaler__scale=[False],
                  union__num_pipeline__poly_features__degree=[1],
                  union__cat_pipeline__manual_dropper__optional_drop_ix = [None],
                  random_forest__n_estimators = [50, 100, 200],
                  random_forest__max_depth = [6, 9, 12],
                  random_forest__max_features = [4, 5, 6])

grid_search_rf = GridSearchCV(estimator=rf_est,
                              param_grid=rf_param_grid,
                              scoring='roc_auc',
                              n_jobs=3,
                              cv=5,
                              refit=True,
                              verbose=1)

grid_search_rf.fit(features_train, outcome_train);

print('Chosen params: {}\n\nTrain AUC score: {:0.3f}'.format(grid_search_rf.best_params_,
                                                        grid_search_rf.best_score_))

#-----------------------------------------------------------
#
# Visualise CV results
#
#-----------------------------------------------------------

rf_results = pd.DataFrame(grid_search_rf.cv_results_)

rf_plt_cols = ['param_random_forest__n_estimators',
               'param_random_forest__max_features',
               'param_random_forest__max_depth',
               'mean_test_score',
               'std_test_score']

rf_ = rf_results.loc[:, rf_plt_cols]
rf_ = rf_.rename(columns = {'param_random_forest__n_estimators':'n_estimators',
                      'param_random_forest__max_depth':'max_depth',
                      'param_random_forest__max_features':'max_features',
                      'mean_test_score' : 'AUC'})
        
train_plot(rf_, f_size = (10,6));

# can access parts of the CV pipeline object like so:
num_pipe_lr = grid_search_lr.best_estimator_.get_params()['union__num_pipeline']
cat_pipe_lr = grid_search_lr.best_estimator_.get_params()['union__cat_pipeline']

# names from numerical pipeline
names_1a = num_pipe_lr.get_params()['zero_var'].get_feature_names(np.array(float_cols + int_cols))
names_2a = num_pipe_lr.get_params()['correlation'].get_feature_names(names_1a)
names_3a = num_pipe_lr.get_params()['poly_features'].get_feature_names(names_2a)

# names from categorical pipeline
names_1b = cat_pipe_lr.get_params()['manual_dropper'].get_feature_names(np.array(fac_col_names))
names_2b = cat_pipe_lr.get_params()['zero_var'].get_feature_names(np.array(names_1b))
names_3b = cat_pipe_lr.get_params()['correlation'].get_feature_names(np.array(names_2b))

# names following final correlation check
names_1c = np.array(names_3a + names_3b.tolist())
lr_feature_names = grid_search_lr.best_estimator_.get_params()['correlation'].get_feature_names(names_1c)

# rebuild features. Notice we selected polynomial terms
# can access parts of the pipeline like so
num_pipe_rf = grid_search_rf.best_estimator_.get_params()['union__num_pipeline']
cat_pipe_rf = grid_search_rf.best_estimator_.get_params()['union__cat_pipeline']

names_1a = num_pipe_rf.get_params()['zero_var'].get_feature_names(np.array(float_cols + int_cols))
names_2a = num_pipe_rf.get_params()['correlation'].get_feature_names(names_1a)
names_3a = num_pipe_rf.get_params()['poly_features'].get_feature_names(names_2a)

names_1b = cat_pipe_rf.get_params()['manual_dropper'].get_feature_names(np.array(fac_col_names))
names_2b = cat_pipe_rf.get_params()['zero_var'].get_feature_names(np.array(names_1b))
names_3b = cat_pipe_rf.get_params()['correlation'].get_feature_names(np.array(names_2b))

names_1c = np.array(names_3a + names_3b.tolist())
rf_feature_names = grid_search_rf.best_estimator_.get_params()['correlation'].get_feature_names(names_1c)

# Lets look at feature importance - relates to how many times the
# feature was chosen for a split in the tree

rf_model = grid_search_rf.best_estimator_.get_params()['random_forest']
rf_importance = rf_model.feature_importances_
sorted(list(zip( rf_importance, rf_feature_names )), key = lambda x: x[0], reverse=True)

# create the test features- use pandas to create 'dummies' for 
# categoricals
fac_dummies_test = pd.get_dummies(features_test.loc[:, fac_cols])
features_test = fac_dummies_test.merge(features_test.loc[:, int_cols + float_cols],
                    left_index = True, right_index = True)

# predictions: 'hard' (i.e. class, based on a 0.5 threshold)
# and probability, i.e. probability positive class
lr_predictions = grid_search_lr.predict(features_test)
lr_prob = grid_search_lr.predict_proba(features_test)[:, 1]

rf_predictions = grid_search_rf.predict(features_test)
rf_prob = grid_search_rf.predict_proba(features_test, )[:, 1]

# intialise object
roc = ROCPlot(outcome_test.values) 
# add predictions from models
roc.calc_roc(rf_prob, 'random forest')
roc.calc_roc(lr_prob, 'logistic regression')
# use buit in helper method for plotting
roc.plot(figsize = (6,6));

print('''Logistic regression AUC: {:0.3f}
Random Forest AUC: {:0.3f}'''.format(roc_auc_score(outcome_test, lr_prob),
                                             roc_auc_score(outcome_test, rf_prob))
     )

# initialise object
lift = LiftChart(outcome_test.values) 
# add the predictions
lift.calc_uplift(rf_prob, 'random forest')
lift.calc_uplift(lr_prob, 'logistic regression')

# built in helper method for plotting
lift.plot(thresh_pct=75, figsize = (6,6));

