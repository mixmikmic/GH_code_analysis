get_ipython().magic('matplotlib inline')
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

os.chdir('C:\\Users\\royal\\Downloads\\Compressed\\data\\d17428d0-e-Criminal')

train=pd.read_csv('criminal_train.csv')

test=pd.read_csv('criminal_test.csv')

##first 5 rows of training data
train.head()

#column names
train.columns

##first 5 rows of testing data
test.head()

##get count of criminal type using groupby function
train.groupby('Criminal')['Criminal'].count()

# for i in train.columns:
#     print(i,train[i].unique())
for i in test.columns:
    print(i,test[i].unique())

# for i in train.columns:
#     print(i,len(train[train[i]==-1]))
for i in test.columns:
    print(i,len(test[test[i]==-1]))

for i in train.columns:
    print(i,train[train[i]==-1].index)

train.drop([19230, 44281],inplace=True)

len(train)

# pd.crosstab(train.POVERTY3,train.Criminal)
pd.crosstab(train.POVERTY3,train.Criminal)

sns.countplot(test.NRCH17_2)

l=['POVERTY3','NRCH17_2']
for i in l:
    train.drop(train[train[i]==-1].index,inplace=True)

##get the perid column and criminal column have to used during creating model
tr_PERID=train['PERID']
ts_Criminal=train['Criminal']

train1=train.drop(['PERID','Criminal'],axis=1)

# train1.head()
test1.head()

test['POVERTY3'].replace(-1,3,inplace=True)
test['NRCH17_2'].replace(-1,0,inplace=True)

ts_PERID=test['PERID']

test1=test.drop(['PERID'],axis=1)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
x,y = train1,ts_Criminal
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)
dd1={}
for i in range(2,10):
    model = KNeighborsClassifier(n_neighbors = i)
    model.fit(x_train,y_train)
    prediction_knn = model.predict(x_test)
    dd1[i]=model.score(x_test,y_test)

dd1

len([i for i in prediction_knn if i==1])

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
x,y = train1,ts_Criminal
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)
model = LinearRegression()
model.fit(x_train,y_train)
prediction_lr = model.predict(x_test)
print(model.score(x_test,y_test))

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
x,y = train1,ts_Criminal
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)
dd={}
for i in range(2,18):
    model = RandomForestClassifier(random_state = i)
    model.fit(x_train,y_train)
    prediction_rfc = model.predict(x_test)
    dd[i]=model.score(x_test,y_test)

dd

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
x,y = train1,ts_Criminal
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)
model = LogisticRegression()
model.fit(x_train,y_train)
prediction_logr = model.predict(x_test)
print(model.score(x_test,y_test))

from sklearn.linear_model import Lasso
from sklearn.cross_validation import train_test_split
x,y = train1,ts_Criminal
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)
model = LogisticRegression()
model.fit(x_train,y_train)
prediction_logr = model.predict(x_test)
print(model.score(x_test,y_test))

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state = 13)
model.fit(train1,ts_Criminal)
prediction_rfc = model.predict(test1)
print(prediction_rfc[:10])

sub = pd.DataFrame({'PERID': ts_PERID, 'Criminal': prediction_rfc})
filename = 'submission.csv'
sub.to_csv(filename, index=False)

df=pd.read_csv('submission.csv')
columnsTitles=["PERID","Criminal"]
df=df.reindex(columns=columnsTitles)
filename = 'submission1.csv'
df.to_csv(filename, index=False)

