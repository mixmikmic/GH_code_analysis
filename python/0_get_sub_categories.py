import pandas as pd
import numpy as np

categories = pd.read_csv('data/categories.csv.gz', compression='gzip', index_col='GROUP_ID')
categories.head()

categories = categories['PATH'].str.split('->', expand=True).rename(columns={0: 'category', 1: 'sub_category'})
categories.head()

categories['category'].unique()

categories['sub_category'].unique()

categories.to_csv('data/categories_parsed.csv.gz', compression='gzip')

train = pd.read_csv('data/evo_train.csv.gz', compression='gzip', index_col='id')
train.head()

data = train.join(categories, on='GROUP_ID')
data.head()

