import pandas as pd
from sklearn import preprocessing
import numpy as np
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
pd.set_option('display.max_columns', 500)

# train = pd.read_csv('../data/clean_train_all_cols_chase.csv', parse_dates=['timestamp'], index_col="id")  
test = pd.read_csv('../data/clean_test_all_cols_chase.csv', parse_dates=['timestamp'], index_col="id")
# locales = pd.read_csv('../data/okurg_chase.csv')
districts = pd.read_csv('../data/district_indices.csv')
# raw = pd.read_csv('../../data/train.csv',parse_dates=['timestamp'])

test.head()

# # get the cols that in both
# tr_cols = train.columns.tolist()
# te_cols = test.columns.tolist()
# cols = set(tr_cols).intersection(te_cols)

# train = train[train.timestamp>'2013-01-01']

# train_index = train.index.tolist()
# test_index = test.index.tolist()

# cols = ['life_sq','full_sq','floor','max_floor','kitch_sq','product_type',
#         'sub_area','kremlin_km','price_doc','timestamp']

# test['price_doc'] = np.nan

# df = pd.concat([train[cols].copy(),
#                 test[cols].copy()],
#                ignore_index=False)

# df['month'] = df.timestamp.dt.month.astype(object)

districts.head()

# df = test[['timestamp','full_sq','okurg_district']]
test['month_year'] = pd.PeriodIndex(test['timestamp'], freq='M').strftime('%m/%y')
districts['month_year'] = pd.PeriodIndex(districts['date'], freq='M').strftime('%m/%y')

districts.columns = ['okurg_district', 'month_year', 'price', 'date', 'nominal_index']

df = df[['month_year','okurg_district','full_sq']]

df.head()

# pd.merge(df,districts[['okurg_district','month_year','price']],how="left", on=["okurg_district","month_year"])

temp = districts[['okurg_district','month_year','price']]

def get_price(x):
    return districts.loc[(districts.okurg_district==x[1]) & (districts.month_year==x[0]),'price']

temp.head()

test = test.merge(temp,how="left",on=['okurg_district','month_year']).set_index(test.index)

test['price_doc'] = test.full_sq * test.price

test.price.hist()
# gold.columns = ['period','month_year','nominal_index']
# macro['month_year'] = pd.PeriodIndex(macro['month_year'], freq='M').strftime('%m/%y')
# df['month_year'] = pd.PeriodIndex(df['timestamp'], freq='M').strftime('%m/%y')

# df = pd.merge(df,gold[['month_year','nominal_index','period']], how="left", on="month_year").reset_index(drop=True).set_index(df.index)

#kaggle 0.49046
test['price_doc'].to_frame().to_csv('../submissions/simple_indices.csv')



