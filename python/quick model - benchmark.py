import pandas as pd
import numpy as np

# preprocessing
from sklearn.preprocessing import LabelEncoder

# modeling
import xgboost as xgb

# validation
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import classification_report

def describe_categorical(X):
    print(X[X.columns[X.dtypes == 'object']].describe())
    
def cstats(y_test, y_test_pred):
    return roc_auc_score(y_test, y_test_pred)

def get_original_datasets(idx):
    global combined
    
    train0 = pd.read_csv('data/train.csv')
    
    targets = train0.Survived
    train = combined.head(idx)
    test = combined.iloc[idx:]
    
    return train, test, targets

def combined_dataset():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    targets = train.Survived
    train.drop('Survived', 1, inplace=True)
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index', inplace=True, axis=1)
    
    return combined, train.shape[0]

combined, idx = combined_dataset()

types = combined.columns.to_series().groupby(combined.dtypes).groups
for k,v in types.items():
    print(k, v)

describe_categorical(combined)

# missing values

combined.isnull().sum()

combined.Age.fillna(combined.Age.mean(), inplace=True)
combined.Fare.fillna(combined.Fare.mean(), inplace=True)

freq_port = combined['Embarked'].mode()[0]
combined.Embarked.fillna(freq_port, inplace=True)

combined['Cabin'] = combined['Cabin'].fillna('X')

combined.isnull().sum()

combined.drop(['Name', 'Ticket', 'Cabin'], inplace=True, axis=1)

combined['Sex'] = combined['Sex'].map({'male':1,'female':0})
le = LabelEncoder()
combined['Embarked'] = le.fit_transform(combined['Embarked'])
combined['Embarked'].head()

types = combined.columns.to_series().groupby(combined.dtypes).groups
for k,v in types.items():
    print(k, v)

train, test, targets = get_original_datasets(idx)

X_train, X_test, y_train, y_test = train_test_split(train, targets, test_size=0.3, random_state=0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

model = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print('training set:', cstats(y_train, y_train_pred))
print('validation set:', cstats(y_test, y_test_pred))

kfold = KFold(n_splits=10, random_state=7)
scores = cross_val_score(model, X_train, y_train, cv=kfold)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print(classification_report(y_test, y_test_pred))



