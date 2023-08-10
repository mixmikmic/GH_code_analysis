import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

get_ipython().magic('matplotlib inline')

df = pd.read_csv('ExtractedFinalDataset.csv')

bad_features = []
for i in range(8):
    langevin = str(i) + "__max_langevin_fixed_point__m_3__r_30"
    bad_features.append(langevin)
    for j in range(9):
        quantile = (j+1)*0.1
        if quantile != 0.5:
            feature_name = str(i) + "__index_mass_quantile__q_" + str(quantile)
            bad_features.append(feature_name)

df = df.drop(bad_features, axis=1)

df.index = df['9']
df = df.drop(['9'], axis=1)
df['Label'] = "One"
df['Label'][2001.0 <= df.index ] = "Two"
df['Label'][4001.0 <= df.index ] = "Three"
df['Label'][6001.0 <= df.index ] = "Four"
df['Label'][8001.0 <= df.index ] = "Five"
df['Label'][10001.0 <= df.index ] = "Six"

df = df[1:]

df.columns = df.columns.map(lambda t: str(t))
df = df.sort_index(axis=1)

extracted_features = df

subsample = extracted_features.sample(frac=0.02).reset_index(drop=True)
subsample.shape

X = subsample.drop(['Label'], 1)
y = subsample['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

import xgboost as xgb
cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}
ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic'}
optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), 
                            cv_params, 
                             scoring = 'accuracy', cv = 5, n_jobs = -1) 

optimized_GBM.fit(X_train, y_train)

optimized_GBM.grid_scores_

model = xgboost.XGBClassifier(learning_rate =0.05,
 n_estimators=3000,
 max_depth=4,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

model.fit(np.array(X_train), np.array(y_train))

print 'Training accuracy:', model.score(np.array(X_train), np.array(y_train))
print 'Test accuracy:', model.score(np.array(X_test), np.array(y_test))

model = xgboost.XGBClassifier(learning_rate =0.05,
 n_estimators=3000,
 max_depth=4,
 min_child_weight=1,
 gamma=0.1,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

model.fit(np.array(X_train), np.array(y_train))

print 'Training accuracy:', model.score(np.array(X_train), np.array(y_train))
print 'Test accuracy:', model.score(np.array(X_test), np.array(y_test))

model = xgboost.XGBClassifier(learning_rate =0.1,
 n_estimators=3000,
 max_depth=4,
 min_child_weight=1,
 gamma=0.1,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

model.fit(np.array(X_train), np.array(y_train))

print 'Training accuracy:', model.score(np.array(X_train), np.array(y_train))
print 'Test accuracy:', model.score(np.array(X_test), np.array(y_test))

model = xgboost.XGBClassifier(learning_rate =0.1,
 n_estimators=3000,
 max_depth=4,
 min_child_weight=1,
 gamma=0.2,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

model.fit(np.array(X_train), np.array(y_train))

print 'Training accuracy:', model.score(np.array(X_train), np.array(y_train))
print 'Test accuracy:', model.score(np.array(X_test), np.array(y_test))

model = xgboost.XGBClassifier(learning_rate =0.2,
 n_estimators=3000,
 max_depth=4,
 min_child_weight=1,
 gamma=0.2,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'multi:softmax',
 nthread=8,
 scale_pos_weight=1,
 seed=27)

model.fit(np.array(X_train), np.array(y_train))

print 'Training accuracy:', model.score(np.array(X_train), np.array(y_train))
print 'Test accuracy:', model.score(np.array(X_test), np.array(y_test))

model = xgboost.XGBClassifier(learning_rate =0.1,
 n_estimators=3000,
 max_depth=7,
 min_child_weight=1,
 gamma=0.2,
 subsample=0.5,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'multi:softmax',
 nthread=8,
 scale_pos_weight=1,
 seed=27)

model.fit(np.array(X_train), np.array(y_train))

print 'Training accuracy:', model.score(np.array(X_train), np.array(y_train))
print 'Test accuracy:', model.score(np.array(X_test), np.array(y_test))

model = xgboost.XGBClassifier(learning_rate =0.1,
 n_estimators=3000,
 max_depth=7,
 min_child_weight=1,
 gamma=0.2,
 subsample=0.5,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'multi:softmax',
 nthread=8,
 scale_pos_weight=1,
 seed=27)

model.fit(np.array(X_train), np.array(y_train))

print 'Training accuracy:', model.score(np.array(X_train), np.array(y_train))
print 'Test accuracy:', model.score(np.array(X_test), np.array(y_test))

model = xgboost.XGBClassifier(learning_rate =0.1,
 n_estimators=3000,
 max_depth=7,
 min_child_weight=1,
 gamma=0.2,
 subsample=0.9,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'multi:softmax',
 nthread=8,
 scale_pos_weight=1,
 seed=27)

model.fit(np.array(X_train), np.array(y_train))

print 'Training accuracy:', model.score(np.array(X_train), np.array(y_train))
print 'Test accuracy:', model.score(np.array(X_test), np.array(y_test))

model = xgboost.XGBClassifier(learning_rate =0.1,
 n_estimators=3000,
 max_depth=7,
 min_child_weight=1,
 gamma=0.2,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.01,
 objective= 'multi:softmax',
 nthread=8,
 scale_pos_weight=1,
 seed=27)

model.fit(np.array(X_train), np.array(y_train))

print 'Training accuracy:', model.score(np.array(X_train), np.array(y_train))
print 'Test accuracy:', model.score(np.array(X_test), np.array(y_test))

model = xgboost.XGBClassifier(learning_rate =0.1,
 n_estimators=3000,
 max_depth=7,
 min_child_weight=1,
 gamma=0.2,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0,
 objective= 'multi:softmax',
 nthread=8,
 scale_pos_weight=1,
 seed=27)

model.fit(np.array(X_train), np.array(y_train))

print 'Training accuracy:', model.score(np.array(X_train), np.array(y_train))
print 'Test accuracy:', model.score(np.array(X_test), np.array(y_test))

model = xgboost.XGBClassifier(learning_rate =0.1,
 n_estimators=1000,
 max_depth=7,
 min_child_weight=1,
 gamma=0.2,
 subsample=0.5,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'multi:softmax',
 nthread=8,
 scale_pos_weight=1,
 seed=27)

model.fit(np.array(X_train), np.array(y_train))

print 'Training accuracy:', model.score(np.array(X_train), np.array(y_train))
print 'Test accuracy:', model.score(np.array(X_test), np.array(y_test))

fullset = extracted_features.sample(frac=1).reset_index(drop=True)
X = fullset.drop(['Label'], 1)
y = fullset['Label']
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X, y, test_size=0.3, random_state=42)

model = xgboost.XGBClassifier(learning_rate =0.05,
 n_estimators=3000,
 max_depth=4,
 min_child_weight=1,
 gamma=0.1,
 subsample=0.5,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'multi:softmax',
 nthread=8,
 scale_pos_weight=1,
 seed=27)

model.fit(np.array(X_train_full), np.array(y_train_full))

print 'Training accuracy:', model.score(np.array(X_train_full), np.array(y_train_full))
print 'Test accuracy:', model.score(np.array(X_test_full), np.array(y_test_full))

from sklearn.metrics import accuracy_score
model = xgboost.XGBClassifier(learning_rate =0.1,
 n_estimators=3000,
 max_depth=4,
 min_child_weight=1,
 gamma=0.1,
 subsample=0.5,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'multi:softmax',
 nthread=8,
 scale_pos_weight=1,
 seed=22)
evaluation_set = [(X_train_full, y_train_full), (X_test_full, y_test_full)]
model.fit(X_train_full, y_train_full, eval_metric=["merror", "mlogloss"], eval_set=evaluation_set,early_stopping_rounds=20, verbose=True)

y_predictions = model.predict(X_test_full)
# evaluate predictions
accuracy = accuracy_score(y_test_full, y_predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# retrieve performance metrics
results = model.evals_result()
epochs = len(results['validation_0']['merror'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.show()
# plot classification error
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['merror'], label='Train')
ax.plot(x_axis, results['validation_1']['merror'], label='Test')
ax.legend()
plt.ylabel('Classification Error')
plt.title('XGBoost Classification Error')
plt.show()

from sklearn.metrics import accuracy_score
model = xgboost.XGBClassifier(learning_rate =0.1,
 n_estimators=3000,
 max_depth=4,
 min_child_weight=1,
 gamma=0,
 subsample=0.5,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'multi:softmax',
 nthread=8,
 scale_pos_weight=1,
 seed=22)
evaluation_set = [(X_train_full, y_train_full), (X_test_full, y_test_full)]
model.fit(X_train_full, y_train_full, eval_metric=["merror", "mlogloss"], eval_set=evaluation_set,early_stopping_rounds=20, verbose=True)

y_predictions = model.predict(X_test_full)
# evaluate predictions
accuracy = accuracy_score(y_test_full, y_predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# retrieve performance metrics
results = model.evals_result()
epochs = len(results['validation_0']['merror'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.show()
# plot classification error
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['merror'], label='Train')
ax.plot(x_axis, results['validation_1']['merror'], label='Test')
ax.legend()
plt.ylabel('Classification Error')
plt.title('XGBoost Classification Error')
plt.show()

