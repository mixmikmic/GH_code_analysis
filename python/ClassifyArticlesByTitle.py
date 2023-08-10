import pandas as pd
import numpy as np

df = pd.read_csv('https://s3-us-west-1.amazonaws.com/simon.bedford/d4d/article_contents.csv')
df = df.fillna('')

df.head()

df.groupby("tag").count()

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

class FSTransformer(BaseEstimator, TransformerMixin):
  """
  Returns the different feature names
  """
  def __init__(self, features):
    self.features = features
    pass

  def fit(self, X, y):
    return self
  
  def transform(self, df):
    return df[self.features].as_matrix()

  
class CountVecTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, col):
    self.vectorizer = CountVectorizer(binary=False)
    self.col = col
    pass
  
  def fit(self, df, y=None):
    self.vectorizer.fit(df[self.col])
    return self
  
  def transform(self, df):
    return self.vectorizer.transform(df[self.col])

cvt = CountVecTransformer("url")
X = cvt.fit_transform(df)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder().fit(df.tag)
y = le.transform(df.tag)

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer

for i in range(5):
  clf_dummy = DummyClassifier(strategy="stratified", random_state=i).fit(X, y)
  print(clf_dummy.score(X, y))

from sklearn.linear_model import RidgeClassifier

clf_ridge = RidgeClassifier().fit(X, y)
print(clf_ridge.score(X, y))

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score

def get_model_scores(model, X, y):
  sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)

  for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index, :], X[test_index, :]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train, y_train)
    r2_train = model.score(X_train, y_train)
    r2_test = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print("R2_train: {0} R2_test: {1} f1: {2}".format(r2_train, r2_test, f1))

get_model_scores(clf_ridge, X, y)

