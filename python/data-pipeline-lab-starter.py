import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# we need to import the template classes to create a class that works like an sklearn class
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

test = pd.read_csv('./assets/datasets/test.csv')

df = pd.read_csv('./assets/datasets/train.csv')
df.head()

df.Age.shape

df.Age.values.reshape(-1,1).shape

# A:
def age_extractor(dataframe):
    df_age_series = dataframe['Age']
    df_age_series.fillna(20, inplace=True)
    return df_age_series.values.reshape(-1,1)

# A:
y = df.Survived

age = age_extractor(df)

# A:
lr = LogisticRegression()
lr.fit(age, y)

test_age = age_extractor(test)

# A:
yhat = lr.predict(test_age)

yhat

# A:
class AgeExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, *args):
        # The transform methods needs to returns some type of data that sklearn can understand
        return X

    def fit(self, X, *args):
        # Fit must return self to work within sklearn pipelines
        return self

# A:

# A:

# A:

# A:

# A:

# A:

# A:

# A:

# A:

# A:

# A:

# A:

# A:

# A:



