import pandas as pd
import numpy as np

train_df = pd.read_csv("../data/ann-train.csv", header=None)
test_df = pd.read_csv("../data/ann-test.csv", header=None)

train_df.head()

test_df.head()

col_names=["age","sex","on thyroxine","query on thyroxine","on antithyroid medication","sick","pregnant","thyroid surgery",
           "I131 treatment","query hypothyroid","query hyperthyroid","lithium","goitre","tumor","hypopituitary","psych",
           "TSH measured","T3 measured","TT4 measured","T4U measured","FTI measured","class"]

train_df.columns = col_names
train_df.head()

test_df.columns = col_names
test_df.head()

train_df.shape

test_df.shape

train_df.info()

test_df.info()

train_df.isnull().values.all()

test_df.isnull().values.all()

train_df['age'] = train_df['age'].apply(lambda x: x*100)
train_df['age'].head()

test_df['age'] = test_df['age'].apply(lambda x: x*100)
test_df['age'].head()

train_df.describe()

test_df.describe()

train_df.to_pickle('../data/train_data_wrangle.plk')
test_df.to_pickle('../data/test_data_wrangle.plk')

