from sml import execute

query = 'READ "../data/computer.csv" (separator = ",", header = 0) AND SPLIT (train = .8, test = .2, validation = .0) AND REGRESS (predictors = [1,2,3,4,5,6,7,8,9], label = 10, algorithm = ridge)'

execute(query, verbose=True)

import pandas as pd
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer

from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from sklearn import linear_model


names = ['vendor name', 'Model Name', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMN', 'CHMAX', 'PRP', 'ERP']

df = pd.read_csv('../data/computer.csv', header = None, names=names)
df.head()

def encode_categorical(df, cols=None):
  categorical = list()
  if cols is not None:
    categorical = cols
  else:
    for col in df.columns:
        if df[col].dtype == 'object':
            categorical.append(col)

  for feature in categorical:
      l = list(df[feature])
      s = set(l)
      l2 = list(s)
      numbers = list()
      for i in range(0,len(l2)):
          numbers.append(i)
      df[feature] = df[feature].replace(l2, numbers)
  return df

df2 =  encode_categorical(df)
df2.head()


features = df2.drop('PRP',1)
label = df2['PRP']
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

ridge = linear_model.Ridge(alpha=0.1)
ridge.fit(X_train, y_train)
pred = ridge.predict(X_test)
r2_score(pred, y_test)



