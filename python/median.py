"Predict sales as a historical median for a given store, day of week, and promo"
"This script scores 0.13888 on the public leaderboard"

import pandas as pd
import numpy as np

train_file = 'data/train.csv'
test_file = 'data/test.csv'
output_file = 'data/predictions.csv'

train = pd.read_csv( train_file )
test = pd.read_csv( test_file )

print train.shape, test.shape

# remove rows with zero sales
# mostly days where closed, but also 54 days when not
train = train.loc[train.Sales > 0]

train.shape

# remove NaNs from Open
test.loc[ test.Open.isnull(), 'Open' ] = 1

train.head()

columns = ['Store', 'DayOfWeek', 'Promo']

medians = train.groupby( columns )['Sales'].median()

medians

medians = medians.reset_index()

medians

test2 = pd.merge( test, medians, on = columns, how = 'left' )
assert( len( test2 ) == len( test ))

test2

test2.loc[ test2.Open == 0, 'Sales' ] = 0
assert( test2.Sales.isnull().sum() == 0 )

test2[[ 'Id', 'Sales' ]].to_csv( output_file, index = False )

print( "Up the leaderboard!" )

columns = ['Store', 'DayOfWeek', 'Promo']

medians = train.groupby( columns )['Customers'].median()

medians = medians.reset_index()

medians

test2 = pd.merge( test, medians, on = columns, how = 'left' )
assert( len( test2 ) == len( test ))

test2.loc[ test2.Open == 0, 'Customers' ] = 0
assert( test2.Customers.isnull().sum() == 0 )

y = train['Sales']
customers = train.pop('Customers')
train.insert(8, 'Customers', customers)
X = train.drop('Sales', axis = 1)
print X.shape, y.shape

train.head()

pd.get_dummies(pd.Series(list('abca')))

np.unique(train.StateHoliday)

np.unique(train.StateHoliday.map(lambda x: str(x)))

pd.get_dummies(train.StateHoliday.map(lambda x: str(x)),"StateHoliday").head()

store = pd.read_csv('data/store.csv')

store.head()



pd.get_dummies(store.PromoInterval, prefix="ProInt", dummy_na = True).head()

alldata = pd.merge(train, store, on='Store', how='left')
alldata.head()

alldata2 = alldata.drop(['Date', 'CompetitionOpenSinceMonth', 'Promo2SinceWeek'], axis = 1)

alldata2.StateHoliday = alldata2.StateHoliday.map(lambda x:str(x))

new = pd.get_dummies(alldata2.StateHoliday, prefix='StateHoliday', dummy_na= True)

alldata3 = pd.concat([alldata2,new],axis=1)

new = pd.get_dummies(alldata2.StoreType, prefix='StoreType', dummy_na= True)

alldata3 = pd.concat([alldata2,new],axis=1)

new = pd.get_dummies(alldata2['Assortment'], prefix='Assortment', dummy_na= True)

alldata3 = pd.concat([alldata2,new],axis=1)

new = pd.get_dummies(alldata2['PromoInterval'], prefix='PromoInterval', dummy_na= True)
alldata3 = pd.concat([alldata2,new],axis=1)

alldata4 = alldata3.drop(['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval'], axis=1)

alldata4 = alldata4.fillna(-1)

y = alldata4.Sales
X = alldata4.drop('Sales', axis=1)

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, accuracy_score, r2_score
etr = ExtraTreesRegressor(n_estimators = 50, oob_score=True, bootstrap = True, warm_start = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
etr.fit(X_train, y_train)
y_pred = etr.predict(X_test)
print 'oob score: ', etr.oob_score_

sorted_mask = np.argsort(etr.feature_importances_)[::-1] #feature importances in descending order

for i in zip(X.columns[sorted_mask], etr.feature_importances_[sorted_mask]):
    print i



