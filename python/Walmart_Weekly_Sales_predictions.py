import pandas as pd
import numpy as np 
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

train = pd.read_csv('C:/DataScience/ML Project/Datasets/Walmart/train.csv')
features = pd.read_csv('C:/DataScience/ML Project/Datasets/Walmart/features.csv')
stores = pd.read_csv('C:/DataScience/ML Project/Datasets/Walmart/stores.csv')

train_m1 = train.merge(features, on=['Store','Date','IsHoliday'],how='left').fillna(0)
train_merge = train_m1.merge(stores, on=['Store'], how='left').fillna(0)
train_data = train_merge.head(len(train_merge) - 450)
#The last 450 rows of data are split off into a validation dataset
train_validate = train_merge.tail(450)

train_data.tail(10)

train_validate

#Since this is weekly sales - the date field is converted into Week of Year.  Also data is standardized as int data with the
#get dummies function
train_data['Date'] = pd.to_datetime(train_data['Date'])
train_data['Year'] = train_data['Date'].dt.year
train_data['Week'] = train_data['Date'].dt.week
train_data['YearWeek'] = train_data.Year.astype(str).str.cat(train_data.Week.astype(str))
train_data.drop(['Date', 'Year', 'Week'], axis=1, inplace=True)
train_data['YearWeek'] = train_data.YearWeek.astype(int)
train_data = pd.get_dummies(train_data)
train_data

y = targets = labels = train_data["Weekly_Sales"].values

columns = ["Store", "Dept", "YearWeek", "IsHoliday", "CPI", "Unemployment", "Size", "Temperature", "Fuel_Price", "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5", "Type_A", "Type_B", "Type_C"]
features = train_data[list(columns)].values
features
# TODO: Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.18, random_state=1)

regressor = DecisionTreeRegressor(max_depth=32, random_state=0)
regressor.fit(X_train, y_train)

regressor.score(X_test, y_test)

list(zip(columns, regressor.feature_importances_))

gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)

gbr.score(X_test, y_test)

list(zip(columns, gbr.feature_importances_))

rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)

rfr.score(X_test, y_test)

list(zip(columns, rfr.feature_importances_))

linreg = LinearRegression()
linreg.fit(X_train, y_train) 

linreg.score(X_test, y_test)

list(zip(columns, linreg.coef_))

bayreg = BayesianRidge(compute_score=True)
bayreg.fit(X_train, y_train)

bayreg.score(X_test, y_test)

list(zip(columns, bayreg.coef_))

train_validate['Date'] = pd.to_datetime(train_validate['Date'])
train_validate['Year'] = train_validate['Date'].dt.year
train_validate['Week'] = train_validate['Date'].dt.week
train_validate['YearWeek'] = train_validate.Year.astype(str).str.cat(train_validate.Week.astype(str))
train_validate.drop(['Date', 'Year', 'Week'], axis=1, inplace=True)
train_validate['YearWeek'] = train_validate.YearWeek.astype(int)
train_validate = pd.get_dummies(train_validate)
train_validate['Type_A'] = 0
train_validate['Type_C'] = 0
train_validate

columns2 = ["Store", "Dept", "YearWeek", "IsHoliday", "CPI", "Unemployment", "Size", "Temperature", "Fuel_Price", "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5", "Type_A", "Type_B", "Type_C"]
features_validate = train_validate[list(columns2)].values
features_validate

pred_vals = rfr.predict(features_validate)

train_validate['Predicted'] = pred_vals
train_validate

train_validate.drop(['Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment', 'Size', 'YearWeek', 'Type_B', 'Type_A', 'Type_C'], axis=1, inplace=True)
train_validate

test = pd.read_csv('C:/DataScience/ML Project/Datasets/Walmart/test.csv')
features = pd.read_csv('C:/DataScience/ML Project/Datasets/Walmart/features.csv')
stores = pd.read_csv('C:/DataScience/ML Project/Datasets/Walmart/stores.csv')
test_m1 = test.merge(features, on=['Store','Date','IsHoliday'],how='left').fillna(0)
test_merge = test_m1.merge(stores, on=['Store'], how='left').fillna(0)

test_merge['Date'] = pd.to_datetime(test_merge['Date'])
test_merge['Year'] = test_merge['Date'].dt.year
test_merge['Week'] = test_merge['Date'].dt.week
test_merge['YearWeek'] = test_merge.Year.astype(str).str.cat(test_merge.Week.astype(str))
test_merge.drop(['Date', 'Year', 'Week'], axis=1, inplace=True)
test_merge['YearWeek'] = test_merge.YearWeek.astype(int)
test_merge = pd.get_dummies(test_merge)
test_merge

columns3 =  ["Store", "Dept", "YearWeek", "IsHoliday", "Size", "Unemployment", "CPI", "Temperature", "Fuel_Price", "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5", "Type_A", "Type_B", "Type_C"]
features_test = test_merge[list(columns2)].values
features_test

test_vals = rfr.predict(features_test)
test_vals

test_merge['PredictedWeeklySales'] = test_vals
test_merge



