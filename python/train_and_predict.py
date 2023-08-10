import numpy as np
import pandas as pd
import lightgbm as lg
from tqdm import tqdm
import gc

# load data
df = pd.read_csv('../train.csv', index_col=0)
df_test = pd.read_csv('../test.csv', index_col=0)

def feature_engineering(data_frame):
    '''
    feature engineering function.
    
    DataFrame -> DataFrame
    '''
    # creating new features
    data_frame['new']  = data_frame['x3B'] - data_frame['x5']
    data_frame['new2']  = data_frame['x3C'] - data_frame['x4']
    data_frame['Day_group_10']  = np.digitize(data_frame.Day, list(range(0,730,10)))
    
    # scalling up "small" features
    small_features_1 = ['x0','x2',"x4"]
    small_features_2 = ["x3A",'x1', "x3B", "x3C", "x3D", "x3E", "x5", "new", "new2"]
    data_frame[small_features_1]= data_frame[small_features_1]*1000
    data_frame[small_features_2]= data_frame[small_features_2]*100000

feature_engineering(df)
feature_engineering(df_test)

# X_train and X_test 
X_train = df.drop(['y','Weight','Day'],1)
X_test = df_test.drop(['Day'],1)
Y = df.y

X_train.head()

# load data into train lightgbm dataset
# notice I'm scaling up the target, making first two columns as categorical features, and load weights
train = lg.Dataset(X_train, Y*10000, categorical_feature=[0, 1], weight=df.Weight, free_raw_data=False)

# hyperparameters for the model
parameters = {'num_leaves': 526, 
 'max_bin': 650, 'feature_fraction': '0.450', 
 'learning_rate': '0.009', 'reg_lambda': 3, 'bagging_freq': 2,
 'min_data_in_leaf': 142, 'colsample_bytree': '0.670', 
 'metric': 'rmse', 'verbose': 1}

boosts = 900
num_ensembles = 15
y_pred = 0.0

# average 15 different models 
for i in tqdm(range(num_ensembles)):
    parameters['seed'] = i * 2332
    model = lg.train(parameters, train_set=train, num_boost_round=boosts + i + 15) 
    y_pred +=  model.predict(data=X_test)
y_pred /= num_ensembles
gc.collect()

yp = pd.Series(y_pred.flatten()/10000).rename('y')
yp.index.name = 'Index'
yp.head()

name = 'model_x'

yp.to_csv('../sub/{}.csv'.format(name), header=True)





