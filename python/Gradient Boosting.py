import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import math
import seaborn as sns
from copy import deepcopy
from collections import defaultdict

sns.set()

# Columns with categorical data that needs to be encoded
categorical_cols = ['serve', 'hitpoint', 'outside.sideline', 'outside.baseline', 'same.side', 
                    'server.is.impact.player', 'outcome', 'gender' , 'previous.hitpoint']

# Columns in the Data That Should Be Scaled
scaled_data = ['rally', 'speed', 'net.clearance', 'distance.from.sideline', 'depth', 
               'player.distance.travelled', 'player.impact.depth', 
               'player.impact.distance.from.center', 'player.depth', 
               'player.distance.from.center', 'previous.speed', 
               'previous.net.clearance', 
               'previous.distance.from.sideline', 'previous.depth', 'opponent.depth', 
               'opponent.distance.from.center', 'previous.time.to.net', 
               'player.impact.distance.from.center']

# Columns to be dropped
train_dropcols = ['id', 'train', 'gender', 'same.side', 'server.is.impact.player', 
                  'outside.baseline', 'previous.hitpoint', 'hitpoint', 'rally', 'serve', 
                  'outside.sideline', 'player.distance.from.center', 'depth', 
                  'player.distance.from.other.length', 'bw.player.distance.penultimate',
                  'player.distance.from.other.width', 'previous.time.to.net',
                  'player.impact.distance.from.center', 'opponent.distance.from.center', 
                  'player.from.net.penultimate','net.clearance.difference',]

# Columns to be dropped
test_dropcols = ['id', 'train', 'gender', 'same.side', 'server.is.impact.player', 'depth',
                 'outside.baseline', 'previous.hitpoint', 'hitpoint', 'rally', 'serve', 
                 'outside.sideline', 'player.distance.from.center', 'previous.time.to.net',
                 'player.distance.from.other.length', 'player.impact.distance.from.center',
                 'player.distance.from.other.width', 'opponent.distance.from.center', 
                 'player.from.net.penultimate', 'bw.player.distance.penultimate',
                 'net.clearance.difference', ]

######################## Load Data #################
raw_mens_train = pd.read_csv('tennis_data/mens_train_file.csv')
raw_mens_test = pd.read_csv("tennis_data/mens_test_file.csv")
raw_womens_train = pd.read_csv('tennis_data/womens_train_file.csv')
raw_womens_test = pd.read_csv("tennis_data/womens_test_file.csv")
raw_mens_train.head()

####################### Feature Engineering #################
def feature_engineer(data):
    
    # Speed difference between previous and last shot
    data['speed.difference'] = data['speed'] - data['previous.speed']
    
    # Net clearance difference between previous and last shot
    data['net.clearance.difference'] = data['net.clearance'] - data['previous.net.clearance']
    
    # The actual distance the player was from the sideline
    data['true.distance.from.sideline'] = [dist if not boolean else (dist * -1) 
                                           for dist, boolean in 
                                           zip(data['distance.from.sideline'], 
                                               data['outside.sideline'])]

    # The actual distance the player was from the baseline
    data['true.distance.from.baseline'] = [dist if not boolean else (dist * -1) 
                                           for dist, boolean in 
                                           zip(data['depth'], 
                                               data['outside.baseline'])]

    # Opponent distance to net + player distance to net
    data['player.distance.from.other.length'] = data['player.depth'] + data['opponent.depth']

    # Opponent distance from center + player distance to center
    data['player.distance.from.other.width'] = [math.fabs(p_center - o_center)
                                                     if boolean else p_center + o_center
                                                     for p_center, o_center, boolean in 
                                                     zip(data['player.distance.from.center'], 
                                                         data['opponent.distance.from.center'], 
                                                         data['same.side'])]

    # Straight line distance between players
    data['bw.player.distance.penultimate'] = [math.hypot(length, width)
                                                 for length, width in 
                                                 zip(data['player.distance.from.other.length'], 
                                                     data['player.distance.from.other.width'])]

    # Distance penultimate shot was made from net- previous to net (s) * penultimate shot (m/s)
    data['player.from.net.penultimate'] = data['previous.speed'] * data['previous.time.to.net']
    
    return data
 
# Perform Feature Engineering
mens_train = feature_engineer(raw_mens_train)
mens_test = feature_engineer(raw_mens_test)
womens_train = feature_engineer(raw_womens_train)
womens_test = feature_engineer(raw_womens_test)
womens_test.head()

##################### Encode Categorical Data ################
def encode(train, test):
    
    # Retain All LabelEncoder as a dictionary
    d = defaultdict(LabelEncoder)
    
    # Encode all the columns
    train[categorical_cols] = train[categorical_cols].apply(lambda x: d[x.name].fit_transform(x))
    test_ids = test['id']

    # Making a deepcopy so we can encode the test data (test data does not have an outcome column)
    temp = deepcopy(categorical_cols)
    temp.remove('outcome')
    e = deepcopy(d)
    del e['outcome']  
    test[temp] = test[temp].apply(lambda x: e[x.name].transform(x))
    
    # Drop the unecessary features
    train = train.drop(train_dropcols, axis=1) 
    test = test.drop(test_dropcols + ['outcome'], axis=1)
    
    return train, test, test_ids, d
    
# Encode Data
mens_train, mens_test, mens_test_ids , mens_dict = encode(mens_train, mens_test)
womens_train, womens_test, womens_test_ids , womens_d = encode(womens_train, womens_test)

mens_train.head()

# Split into inputs and output 
mens_train_X = mens_train.loc[:, mens_train.columns != 'outcome']
mens_train_y = mens_train['outcome']
womens_train_X = womens_train.loc[:, womens_train.columns != 'outcome']
womens_train_y = womens_train['outcome']

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

mens_xgb_model = XGBClassifier(n_estimators=100, random_state= 1, learning_rate= 0.1, 
                               subsample= 0.9, colsample_bytree= 1.0, 
                               eval_metric= 'mlogloss', max_depth= 5, 
                               min_child_weight= 1, gamma= 0)

y_pred = cross_val_predict(mens_xgb_model, mens_train_X, mens_train_y,cv=4)
confusion_matrix(mens_train_y,y_pred)

from sklearn.model_selection import cross_val_score

CV = 4

###### Men's Model #######
mens_xgb_model = XGBClassifier(n_estimators=100, random_state= 1, learning_rate= 0.1, 
                               subsample= 0.9, colsample_bytree= 1.0, max_depth= 5, 
                               eval_metric= 'mlogloss', min_child_weight= 1)
m_loss = cross_val_score(mens_xgb_model, mens_train_X, mens_train_y, 
                         scoring='neg_log_loss', cv = CV, n_jobs=-1)

m_acc = cross_val_score(mens_xgb_model, mens_train_X, mens_train_y, 
                        scoring='accuracy', cv = CV, n_jobs=-1)


###### Womens's Model #######
womens_xgb_model = XGBClassifier(n_estimators=100, random_state= 1, learning_rate= 0.1, 
                                 subsample= 0.9, colsample_bytree= 0.8, max_depth= 5, 
                                 eval_metric= 'mlogloss', min_child_weight= 1)
w_loss = cross_val_score(womens_xgb_model, womens_train_X, womens_train_y, 
                         scoring='neg_log_loss', cv = CV, n_jobs=-1)
w_acc = cross_val_score(womens_xgb_model, womens_train_X, womens_train_y, 
                        scoring='accuracy', cv = CV, n_jobs=-1)

print("Number of k-folds: " + str(CV))
print("---------------------------------")

print("Men's Model Results")
print("Accuracy: {:.4%}".format(np.mean(m_acc)))
print("Log Loss: {}".format(np.mean(m_loss)))

print("---------------------------------")

print("Women's Model Results")
print("Accuracy: {:.4%}".format(np.mean(w_acc)))
print("Log Loss: {}".format(np.mean(w_loss)))

# Create new training and validation set using scikit's train_test_split
mens_train_XX, val_mens_XX = train_test_split(mens_train, test_size=0.2, shuffle=True)
mens_X_train = mens_train_XX.loc[:, mens_train_XX.columns != 'outcome']
mens_y_train = mens_train_XX['outcome']
mens_X_val = val_mens_XX.loc[:, val_mens_XX.columns != 'outcome']
mens_y_val = val_mens_XX['outcome']

mens_model = XGBClassifier(eval_metric='mlogloss')
mens_model.fit(mens_X_train, mens_y_train)
mens_y_prob_pred = mens_model.predict_proba(mens_X_val)
mens_y_pred = mens_model.predict(mens_X_val)
mens_loss = log_loss(mens_y_val, mens_y_prob_pred)
mens_acc = accuracy_score(mens_y_val, mens_y_pred)
print("Accuracy: {:.4%}".format(mens_acc))
print("Log Loss: {}".format(mens_loss))
print('----------------')

for x in zip(mens_X_train,mens_model.feature_importances_):
    print(x)

plt.bar(range(len(mens_model.feature_importances_)), mens_model.feature_importances_)
plt.show()

from sklearn.model_selection import GridSearchCV

# n_estimators = [100, 200, 300, 400, 500]
# learning_rate = [0.0001, 0.001, 0.01, 0.1]

cv_params = {'learning_rate': [0.001, 0.01, 0.1]}
ind_params = {'n_estimators': 100, 'seed':0,  'subsample': 0.9, 'colsample_bytree':1.0,
              'objective': 'multi:softprob', 'max_depth': 5, 'min_child_weight': 1}
optimized_GBM = GridSearchCV(XGBClassifier(**ind_params), 
                             cv_params, scoring = 'neg_log_loss', cv = 4, n_jobs = -1, verbose=3) 

optimized_GBM.fit(mens_train_X, mens_train_y)
print('Optimzed Scores')
optimized_GBM.grid_scores_

# Train Men's Model and Make Predictions
mens_model = XGBClassifier(n_estimators=100, seed= 0, learning_rate= 0.1, subsample= 0.9, 
                               colsample_bytree= 0.8, eval_metric= 'mlogloss', max_depth= 5, 
                               min_child_weight= 1, gamma= 0)

mens_model.fit(mens_train_X, mens_train_y)
mens_test_pred = pd.DataFrame(mens_model.predict_proba(mens_test))


# Train Women's Model and Make Predictions
womens_model = XGBClassifier(n_estimators=100, seed= 0, learning_rate= 0.1, subsample= 0.9, 
                               colsample_bytree= 0.8, eval_metric= 'mlogloss', max_depth= 5, 
                               min_child_weight= 1, gamma= 0)

womens_model.fit(womens_train_X, womens_train_y)
womens_test_pred = pd.DataFrame(womens_model.predict_proba(womens_test))

def append_gender(data, gender):
    return str(data) + '_' + str(gender)

def create_column_ids(mens_id, womens_id):
    mens_test_id_col = mens_id.apply(append_gender, args=('mens',))
    womens_test_id_col = womens_id.apply(append_gender, args=('womens',))

    combined_id = np.concatenate((mens_test_id_col, womens_test_id_col))
    
    return pd.DataFrame(combined_id)

column_ids = create_column_ids(mens_test_ids, womens_test_ids)

combined_test_predictions = pd.concat([mens_test_pred, womens_test_pred], axis=0)
combined_test_predictions.columns = ['FE', 'UE', 'W']
combined_test_predictions.reset_index(inplace=True, drop=True)
combined_test_predictions.head()

import time

def format_submission(predictions):
    format_file = pd.read_csv('tennis_data/AUS_SubmissionFormat.csv')
    final = pd.concat([column_ids, format_file[['train']], predictions], axis=1, )
    final.columns = ['submission_id', 'train', 'FE', 'UE', 'W']
    final = final.set_index(list(final[['submission_id']])).T
    correct_order = list(format_file['submission_id'])
    final_sorted = final[correct_order].T.reset_index()
    cols = ['submission_id', 'train', 'UE', 'FE', 'W']
    final_sorted = final_sorted[cols]
    return final_sorted

final_submission = format_submission(combined_test_predictions)
final_submission.head()

# Save file with timestamp
timestr = time.strftime("%Y%m%d-%H%M%S")
def save_file(data):
    data.to_csv('Submissions/SubmissionGB' + timestr + '.csv', index=False)

save_file(final_submission)

