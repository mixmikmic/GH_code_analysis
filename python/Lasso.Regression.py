import pandas as pd
import numpy as np
from sklearn import linear_model

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int,
              'sqft_living15':float, 'grade':int, 'yr_renovated':int,
              'price':float, 'bedrooms':float, 'zipcode':str, 'long':float,
              'sqft_lot15':float, 'sqft_living':float, 'floors':float,
              'condition':int, 'lat':float, 'date':str, 'sqft_basement':int,
              'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales = pd.read_csv("kc_house_data.csv", dtype=dtype_dict)

from math import log, sqrt
sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']

# In the dataset, 'floors' was defined with type string, 
# so we'll convert them to float, before creating a new feature.
sales['floors'] = sales['floors'].astype(float) 
sales['floors_square'] = sales['floors']*sales['floors']

all_features = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']

model_all = linear_model.Lasso(alpha=5e2, normalize=True) # set parameters
model_all.fit(sales[all_features], sales['price']) # learn weights

for i in  range(len(model_all.coef_)):
    if model_all.coef_[i] != 0:
        print sales[all_features].columns.values[i]

testing = pd.read_csv('wk3_kc_house_test_data.csv', dtype=dtype_dict)
training = pd.read_csv('wk3_kc_house_train_data.csv', dtype=dtype_dict)
validation = pd.read_csv('wk3_kc_house_valid_data.csv', dtype=dtype_dict)

testing['sqft_living_sqrt'] = testing['sqft_living'].apply(sqrt)
testing['sqft_lot_sqrt'] = testing['sqft_lot'].apply(sqrt)
testing['bedrooms_square'] = testing['bedrooms']*testing['bedrooms']
testing['floors_square'] = testing['floors']*testing['floors']

training['sqft_living_sqrt'] = training['sqft_living'].apply(sqrt)
training['sqft_lot_sqrt'] = training['sqft_lot'].apply(sqrt)
training['bedrooms_square'] = training['bedrooms']*training['bedrooms']
training['floors_square'] = training['floors']*training['floors']

validation['sqft_living_sqrt'] = validation['sqft_living'].apply(sqrt)
validation['sqft_lot_sqrt'] = validation['sqft_lot'].apply(sqrt)
validation['bedrooms_square'] = validation['bedrooms']*validation['bedrooms']
validation['floors_square'] = validation['floors']*validation['floors']

best_l1_penalty = None
best_RSS = None

for l1_penalty in np.logspace(1, 7, num=13):
    lasso = linear_model.Lasso(alpha = l1_penalty, normalize=True)
    lasso.fit(training[all_features], training['price'])
    Y_predicted = lasso.predict(validation[all_features])
    RSS = ((Y_predicted - validation['price'])**2).sum()
    
    if best_RSS is None or best_RSS > RSS:
        best_RSS = RSS
        best_l1_penalty = l1_penalty

print best_l1_penalty
print best_RSS

model = linear_model.Lasso(alpha=best_l1_penalty, normalize=True) # set parameters
model.fit(training[all_features], training['price'])# learn weights
np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_)

max_nonzeros = 7

l1_penalty_values = np.logspace(1, 4, num=20)

l1_penalty_list = []
RSS_values = []
nz_coef = []

for l1_penalty in l1_penalty_values:
    RSS = 0
    non_zeroes_coef = 0
    lasso = linear_model.Lasso(alpha = l1_penalty, normalize=True)
    lasso.fit(training[all_features], training['price'])
    Y_predicted = lasso.predict(validation[all_features])
    RSS = ((Y_predicted - validation['price'])**2).sum()
    non_zeroes_coef = np.count_nonzero(lasso.coef_) + np.count_nonzero(lasso.intercept_)
    
    l1_penalty_list.append(l1_penalty)
    RSS_values.append(RSS)
    nz_coef.append(non_zeroes_coef)

l1_data = pd.DataFrame({'l1_penalty_list':l1_penalty_list,
                        'RSS_values':RSS_values,
                        'nz_coef':nz_coef})

l1_data

l1_penalty_min = l1_data[l1_data['nz_coef'] > max_nonzeros]['l1_penalty_list'].max()
l1_penalty_max = l1_data[l1_data['nz_coef'] < max_nonzeros]['l1_penalty_list'].min()

print l1_penalty_min
print l1_penalty_max

l1_penalty_values = np.linspace(l1_penalty_min,l1_penalty_max,20)

best_l1_penalty = None
best_RSS = None

for l1_penalty in l1_penalty_values:
    lasso = linear_model.Lasso(alpha = l1_penalty, normalize=True)
    lasso.fit(training[all_features], training['price'])
    Y_predicted = lasso.predict(validation[all_features])
    RSS = ((Y_predicted - validation['price'])**2).sum()
    non_zeroes_coef = np.count_nonzero(lasso.coef_) + np.count_nonzero(lasso.intercept_)
    
    if non_zeroes_coef == max_nonzeros:
        if best_RSS is None or best_RSS > RSS:
            best_RSS = RSS
            best_l1_penalty = l1_penalty

best_l1_penalty

lasso = linear_model.Lasso(alpha = best_l1_penalty, normalize=True)
lasso.fit(training[all_features], training['price'])

for i in  range(len(lasso.coef_)):
    if lasso.coef_[i] != 0:
        print training[all_features].columns.values[i]



