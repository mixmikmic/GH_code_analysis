import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import sklearn.model_selection
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import confusion_matrix, mean_squared_error
import matplotlib
get_ipython().magic('matplotlib inline')

from slackclient import SlackClient
def slack_message(message, channel):
    token = 'your_token'
    sc = SlackClient(token)
    sc.api_call('chat.postMessage', channel=channel, 
                text=message, username='My Sweet Bot',
                icon_emoji=':upside_down_face:')

data_dir = '/your/directory/'  
data_file = data_dir + 'data_file'

data = pd.read_csv(data_file, sep="\t", parse_dates = ['dates'], infer_datetime_format = True)

def replaceValues(df):
    df.replace(r'[\s]','_', inplace = True, regex = True)
    df.replace(r'[\.]','', inplace = True, regex = True)
    df.replace(r'__','_', inplace = True, regex = True)

replaceValues(data)

cat_cols = ['ATTRIBUTE_1','ATTRIBUTE_2','ATTRIBUTE_3']
index_cols = ['FACTOR_1','FACTOR_2','FACTOR_3']
pred_cols = ['RESPONSE']

num_cols = [x for x in list(data.columns.values) if x not in cat_cols if x not in fac_cols if x not in pred_cols]

# def categoricalCols(indf, cat_var_list):
#     for cv in cat_var_list:
#         if [i for i, x in enumerate(cat_var_list) if cv == x][0] == 0:
#             dummy_df = pd.get_dummies(indf[cv], prefix = cv)
#         else:
#             dummy_df = pd.concat([dummy_df, pd.get_dummies(indf[cv], prefix = cv)], axis = 1)
#     return dummy_df
        
# combined_cat = categoricalCols(combined[cat_cols], cat_cols)
# combined_cat.columns.values

data_cat = pd.DataFrame(data[cat_cols])
for feature in cat_cols: # Loop through all columns in the dataframe
    if data_cat[feature].dtype == 'object': # Only apply for columns with categorical strings
        data_cat[feature] = pd.Categorical(data[feature]).codes # Replace strings with an integer

data_num = data[num_cols]
data_final = pd.concat([data_cat, data_num], axis=1)
data_final['DATE'] = data['DATE']
data_final['RESPONSE'] = data['RESPONSE']
print data_final.shape

train_final = data_final[data_final['DATE'] <= 'DATE_SPLIT']
test_final = data_final[data_final['DATE'] >= 'DATE_SPLIT' ]

print(train_final.shape)
print(test_final.shape)

train = data[data['DATE'] <= 'DATE_SPLIT']
test = data[data['DATE'] >= 'DATE_SPLIT' ]

print(train.shape)
print(test.shape)

y_train = train_final['RESPONSE']
y_test = test_final['RESPONSE']
x_train = train_final.drop(['RESPONSE','DATE'], axis=1)
x_test = test_final.drop(['RESPONSE','DATE'], axis=1)

print x_train.columns.values

objective = "reg:linear"
seed = 100
n_estimators = 100
learning_rate = 0.1
gamma = 0.1
subsample = 0.8
colsample_bytree = 0.8
reg_alpha = 1
reg_lambda = 1
silent = False

parameters = {}
parameters['objective'] = objective
parameters['seed'] = seed
parameters['n_estimators'] = n_estimators
parameters['learning_rate'] = learning_rate
parameters['gamma'] = gamma
parameters['colsample_bytree'] = colsample_bytree
parameters['reg_alpha'] = reg_alpha
parameters['reg_lambda'] = reg_lambda
parameters['silent'] = silent

scores = []

cv_params = {'max_depth': [2,4,6,8],
             'min_child_weight': [1,3,5,7]
            }

gbm = GridSearchCV(xgb.XGBRegressor(
                                        objective = objective,
                                        seed = seed,
                                        n_estimators = n_estimators,
                                        learning_rate = learning_rate,
                                        gamma = gamma,
                                        subsample = subsample,
                                        colsample_bytree = colsample_bytree,
                                        reg_alpha = reg_alpha,
                                        reg_lambda = reg_lambda,
                                        silent = silent

                                    ),
                    
                    param_grid = cv_params,
                    iid = False,
                    scoring = "neg_mean_squared_error",
                    cv = 5,
                    verbose = True
)

gbm.fit(x_train,y_train)
print gbm.cv_results_
print "Best parameters %s" %gbm.best_params_
print "Best score %s" %gbm.best_score_
slack_message("max_depth and min_child_weight parameters tuned! moving on to refinement", 'channel')

max_depth = gbm.best_params_['max_depth']
min_child_weight = gbm.best_params_['min_child_weight']
parameters['max_depth'] = max_depth
parameters['min_child_weight'] = min_child_weight
scores.append(gbm.best_score_)

cv_params = {'max_depth': [max_depth-1, max_depth, max_depth+1], 
             'min_child_weight': [min_child_weight-1, min_child_weight-0.5, min_child_weight, min_child_weight+0.5, min_child_weight+1]
            }

gbm = GridSearchCV(xgb.XGBRegressor(
                                        objective = objective,
                                        seed = seed,
                                        n_estimators = n_estimators,
                                        learning_rate = learning_rate,
                                        gamma = gamma,
                                        subsample = subsample,
                                        colsample_bytree = colsample_bytree,
                                        reg_alpha = reg_alpha,
                                        reg_lambda = reg_lambda,
                                        silent = silent

                                    ),
                   
                    param_grid = cv_params,
                    iid = False,
                    scoring = "neg_mean_squared_error",
                    cv = 5,
                    verbose = True
)

gbm.fit(x_train,y_train)
print gbm.cv_results_
print "Best parameters %s" %gbm.best_params_
print "Best score %s" %gbm.best_score_
slack_message("max_depth and min_child_weight parameters refined! moving on to tuning gamma parameter", 'channel')

max_depth = gbm.best_params_['max_depth']
min_child_weight = gbm.best_params_['min_child_weight']
parameters['max_depth'] = max_depth
parameters['min_child_weight'] = min_child_weight
scores.append(gbm.best_score_)

cv_params = {'gamma': [i/10.0 for i in range(1,10,2)]}

gbm = GridSearchCV(xgb.XGBRegressor(
                                        objective = objective,
                                        seed = seed,
                                        n_estimators = n_estimators,
                                        max_depth = max_depth,
                                        min_child_weight = min_child_weight,
                                        learning_rate = learning_rate,
                                        subsample = subsample,
                                        colsample_bytree = colsample_bytree,
                                        reg_alpha = reg_alpha,
                                        reg_lambda = reg_lambda,
                                        silent = silent

                                    ),
                   
                    param_grid = cv_params,
                    iid = False,
                    scoring = "neg_mean_squared_error",
                    cv = 5,
                    verbose = True
)

gbm.fit(x_train,y_train)
print gbm.cv_results_
print "Best parameters %s" %gbm.best_params_
print "Best score %s" %gbm.best_score_
slack_message("gamma tuned! moving on to tuning subsample and colsample_bytree parameters", 'channel')

gamma = gbm.best_params_['gamma']
parameters['gamma'] = gamma
scores.append(gbm.best_score_)

cv_params = {'subsample': [i/10.0 for i in range(6,11)],
             'colsample_bytree': [i/10.0 for i in range(6,11)]
            }

gbm = GridSearchCV(xgb.XGBRegressor(
                                        objective = objective,
                                        seed = seed,
                                        n_estimators = n_estimators,
                                        max_depth = max_depth,
                                        min_child_weight = min_child_weight,
                                        learning_rate = learning_rate,
                                        gamma = gamma,
                                        reg_alpha = reg_alpha,
                                        reg_lambda = reg_lambda,
                                        silent = silent

                                    ),
                   
                    param_grid = cv_params,
                    iid = False,
                    scoring = "neg_mean_squared_error",
                    cv = 5,
                    verbose = True
)

gbm.fit(x_train,y_train)
print gbm.cv_results_
print "Best parameters %s" %gbm.best_params_
print "Best score %s" %gbm.best_score_
slack_message("subsample and colsample_bytree parameters tuned! moving on to refinement", 'channel')

subsample = gbm.best_params_['subsample']
colsample_bytree = gbm.best_params_['colsample_bytree']
parameters['subsample'] = subsample
parameters['colsample_bytree'] = colsample_bytree
scores.append(gbm.best_score_)

cv_params = {'subsample': [i/100.0 for i in range(int((subsample-0.1)*100.0), min(int((subsample+0.1)*100),105) , 5)],
             'colsample_bytree': [i/100.0 for i in range(int((colsample_bytree-0.1)*100.0), min(int((subsample+0.1)*100),105), 5)]
            }

gbm = GridSearchCV(xgb.XGBRegressor(
                                        objective = objective,
                                        seed = seed,
                                        n_estimators = n_estimators,
                                        max_depth = max_depth,
                                        min_child_weight = min_child_weight,
                                        learning_rate = learning_rate,
                                        gamma = gamma,
                                        reg_alpha = reg_alpha,
                                        reg_lambda = reg_lambda,
                                        silent = silent

                                    ),
                   
                    param_grid = cv_params,
                    iid = False,
                    scoring = "neg_mean_squared_error",
                    cv = 5,
                    verbose = True
)

gbm.fit(x_train,y_train)
print gbm.cv_results_
print "Best parameters %s" %gbm.best_params_
print "Best score %s" %gbm.best_score_
slack_message("subsample and colsample_bytree parameters refined! moving on to tuning the alpha and lambda parameters", 'channel')

colsample_bytree = gbm.best_params_['colsample_bytree']
subsample = gbm.best_params_['subsample']
parameters['colsample_bytree'] = colsample_bytree
parameters['subsample'] = subsample
scores.append(gbm.best_score_)

cv_params = {'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100], 
             'reg_lambda': [1e-5, 1e-2, 0.1, 1, 100]
            }

gbm = GridSearchCV(xgb.XGBRegressor(
                                        objective = objective,
                                        seed = seed,
                                        n_estimators = n_estimators,
                                        max_depth = max_depth,
                                        min_child_weight = min_child_weight,
                                        learning_rate = learning_rate,
                                        gamma = gamma,
                                        colsample_bytree = colsample_bytree,
                                        subsample = subsample,
                                        silent = silent

                                    ),
                   
                    param_grid = cv_params,
                    iid = False,
                    scoring = "neg_mean_squared_error",
                    cv = 5,
                    verbose = True
)

gbm.fit(x_train,y_train)
print gbm.cv_results_
print "Best parameters %s" %gbm.best_params_
print "Best score %s" %gbm.best_score_
slack_message("alpha and lambda parameters tuned! moving on to refinement", 'channel')

reg_alpha = gbm.best_params_['reg_alpha']
reg_lambda = gbm.best_params_['reg_lambda']
parameters['reg_alpha'] = reg_alpha
parameters['reg_lambda'] = reg_lambda
scores.append(gbm.best_score_)

cv_params = {'reg_lambda': [reg_alpha*0.2, reg_alpha*0.5, reg_alpha, reg_alpha*2, reg_alpha*5], 
             'reg_alpha': [reg_lambda*0.2, reg_lambda*0.5, reg_lambda, reg_lambda*2, reg_lambda*5]
            }

gbm = GridSearchCV(xgb.XGBRegressor(
                                        objective = objective,
                                        seed = seed,
                                        n_estimators = n_estimators,
                                        max_depth = max_depth,
                                        min_child_weight = min_child_weight,
                                        learning_rate = learning_rate,
                                        gamma = gamma,
                                        colsample_bytree = colsample_bytree,
                                        subsample = subsample,
                                        silent = silent

                                    ),
                   
                    param_grid = cv_params,
                    iid = False,
                    scoring = "neg_mean_squared_error",
                    cv = 5,
                    verbose = True
)

gbm.fit(x_train,y_train)
print gbm.cv_results_
print "Best parameters %s" %gbm.best_params_
print "Best score %s" %gbm.best_score_
slack_message("alpha and lambda parameters refined! finalising model by reducing learning rate and increasing trees", 'channel')

reg_alpha = gbm.best_params_['reg_alpha']
reg_lambda = gbm.best_params_['reg_lambda']
parameters['reg_alpha'] = reg_alpha
parameters['reg_lambda'] = reg_lambda
scores.append(gbm.best_score_)

print parameters
print scores

# n_estimators = 3000
# learning_rate = 0.05

# parameters['n_estimators'] = n_estimators
# parameters['learning_rate'] = learning_rate

# xgbFinal = xgb.XGBRegressor(
#     objective = objective,
#     seed = seed,
#     n_estimators = n_estimators,
#     max_depth = max_depth,
#     min_child_weight = min_child_weight,
#     learning_rate = learning_rate,
#     gamma = gamma,
#     subsample = subsample,
#     colsample_bytree = colsample_bytree,
#     reg_alpha = reg_alpha,
#     reg_lambda = reg_lambda,
#     silent = False
# )

# xgb1.fit(x_train, y_train, eval_set = [(x_train, y_train), (x_test, y_test)], eval_metric = 'rmse', verbose = True)
# slack_message("Training complete!", 'channel')

trainDMat = xgb.DMatrix(data = x_train, label = y_train)
testDMat = xgb.DMatrix(data = x_test, label = y_test)

learning_rate = 0.05
parameters['eta'] = learning_rate

num_boost_round = 3000
early_stopping_rounds = 20

xgbCV = xgb.cv(
    params = parameters, 
    dtrain = trainDMat, 
    num_boost_round = num_boost_round,
    nfold = 5,
    metrics = {'rmse'},
    early_stopping_rounds = early_stopping_rounds,
    verbose_eval = True,
    seed = seed     
)

slack_message("Training complete! Producing final booster object", 'channel')

num_boost_round = len(xgbCV)
parameters['eval_metric'] = 'rmse'

xgbFinal = xgb.train(
    params = parameters, 
    dtrain = trainDMat, 
    num_boost_round = num_boost_round,
    evals = [(trainDMat, 'train'), 
             (testDMat, 'eval')]
)

slack_message("Booster object created!", 'channel')

xgb.plot_importance(xgbFinal)

xgbFinal_train_preds = xgbFinal.predict(x_train)
xgbFinal_test_preds = xgbFinal.predict(x_test)

print(xgbFinal_train_preds.shape)
print(xgbFinal_test_preds.shape)

print "\nModel Report"
print "MSE Train : %f" % mean_squared_error(y_train, xgbFinal_train_preds)
print "MSE Test: %f" % mean_squared_error(y_test, xgbFinal_test_preds)
print "RMSE Train: %f" % mean_squared_error(y_train, xgbFinal_train_preds)**0.5
print "RMSE Test: %f" % mean_squared_error(y_test, xgbFinal_test_preds)**0.5

pickle.dump(xgbFinal, open("xgbFinal.pickle.dat", "wb"))

# xgb1 = pickle.load(open("xgb1.pickle.dat", "rb"))
# xgb1_train_preds = xgb1.predict(x_train)
# xgb1_test_preds = xgb1.predict(x_test)

# print "\nModel Report"
# print "MSE Train : %f" % mean_squared_error(y_train, xgb1_train_preds)
# print "MSE Test: %f" % mean_squared_error(y_test, xgb1_test_preds)
# print "RMSE Train: %f" % mean_squared_error(y_train, xgb1_train_preds)**0.5
# print "RMSE Test: %f" % mean_squared_error(y_test, xgb1_test_preds)**0.5

train_preds = pd.DataFrame(xgbFinal_train_preds)
test_preds = pd.DataFrame(xgbFinal_test_preds)
train_preds.columns = ['RESPONSE']
test_preds.column = ['RESPONSE']

train.to_csv('XGBoost Train.csv', sep=',')
train_preds.to_csv('XGBoost Train Preds.csv', sep=',')
test.to_csv('XGBoost Test.csv', sep=',')
test_preds.to_csv('XGBoost Test Preds.csv', sep=',')
slack_message("Files saved!", 'channel')

