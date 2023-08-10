import numpy as np
import sklearn 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
get_ipython().magic('matplotlib inline')

train1 = pd.read_excel("datasets/dataset1.xlsx")
train2 = pd.read_excel("datasets/dataset2.xlsx")

train1.columns

train2.columns

train2 = train2.drop(['Row Number'],axis=1)
train1 = train1.drop(['Row Number'],axis=1)

train = pd.concat([train1,train2]).reset_index()
train.head()

# percentage of 0s and 1s
(train.Outcome.value_counts()/750)*100

train.Outcome.loc[:,].hist()

train.isnull().sum()

plt.scatter(x=train.iloc[:,1:2],y=train.iloc[:,1:2])

# prepare values to be considered by the model.
# X includes all the features Fico Score	Length of Employment (Months)	Length of Current Residency (Months)	Monthly Income	Debt to Income percentage	
X = train.iloc[:,1:-1].values

X

# Response labels
y = train.iloc[:,6:].values

y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_test

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

from xgboost import XGBClassifier
model_xgb = XGBClassifier(n_estimators=5,learning_rate=0.02)

y_train[0]

model_xgb.fit(X_train,y_train.ravel())

predictions = model_xgb.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)

from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators=1000)
model_rf.fit(X_train,y_train.ravel())

predictions_rf = model_rf.predict(X_test)

accuracy_score(y_test,predictions_rf)

from sklearn.linear_model import LassoCV
model_lasso = LassoCV(eps=0.0001, n_alphas=1000)
model_lasso.fit(X_train,y_train.ravel())

predictions_lasso = model_lasso.predict(X_test)
predictions_lasso = predictions_lasso>0.5

accuracy_score(y_test,predictions_lasso)

from sklearn.linear_model import ElasticNet
model_enet = ElasticNet()
model_enet.fit(X_train,y_train.ravel())

predictions_enet = model_enet.predict(X_test)
predictions_enet = predictions_enet>0.5

accuracy_score(y_test,predictions_enet)

from sklearn.linear_model import LogisticRegression
model_logregression = LogisticRegression()
model_logregression.fit(X_train,y_train.ravel())

predictions_logregression = model_logregression.predict(X_test)
predictions_logregression

accuracy_score(y_test,predictions_logregression)

from sklearn.svm import SVC
model_svc = SVC(C=1,kernel='rbf')
model_svc.fit(X_train,y_train.ravel())

predictions_svc = model_svc.predict(X_test)
predictions_svc

accuracy_score(y_test,predictions_svc)

model_lgb = lgb.LGBMClassifier(num_leaves=122,learning_rate=0.0001)
model_lgb.fit(X_train,y_train.ravel())

predictions_lgb = model_lgb.predict(X_test)
predictions_lgb

accuracy_score(y_test,predictions_lgb)

#Validation function
n_folds = 5

def cross_score(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    return cross_val_score(model, X, y.ravel(), scoring="accuracy", cv = kf)    

def model_performance(model):
    scores = cross_score(model)
    print(scores)
    print("\n")
    print("Mean:{}, Standard Deviation:{}".format(np.mean(scores),np.std(scores)))

# How does logistic ression stay along for predictions?
model_performance(model_logregression)

# How does xgboost stay along for predictions?
model_performance(model_xgb)

# How does SVC stay along for predictions?
model_performance(model_svc)

# How does lgb stay along for predictions?# How does xgboost stay along for predictions?
model_performance(model_lgb)



