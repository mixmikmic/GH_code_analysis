import graphlab

sales = graphlab.SFrame('kc_house_data.gl/')

from math import log, sqrt
sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']

# In the dataset, 'floors' was defined with type string, 
# so we'll convert them to int, before creating a new feature.
sales['floors'] = sales['floors'].astype(int) 
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
len(all_features)

model_all = graphlab.linear_regression.create(sales, target='price', features=all_features,
                                              validation_set=None, 
                                              l2_penalty=0., l1_penalty=1e10, verbose=False)

model_all.get('coefficients').print_rows(num_rows=len(all_features) + 1)

(training_and_validation, testing) = sales.random_split(.9,seed=1) # initial train/test split
(training, validation) = training_and_validation.random_split(0.5, seed=1) # split training into train and validate

import numpy as np
l1_penalty_list = np.logspace(1, 7, num=13)
sort_table = []

for l1_penalty in l1_penalty_list:
    model = graphlab.linear_regression.create(training, target='price', features=all_features,
                                              validation_set=None, 
                                              l2_penalty=0, l1_penalty=l1_penalty, verbose=False)
    predictions = model.predict(validation)
    RSS = sum([(predictions[i] - validation[i]['price']) ** 2 for i in range(len(predictions))])
    print l1_penalty, RSS
    sort_table.append((RSS, l1_penalty))
    
print sorted(sort_table)[0]

best_model = graphlab.linear_regression.create(training, target='price', features=all_features,
                                              validation_set=None, 
                                              l2_penalty=0, l1_penalty=10, verbose=False)
best_model.get('coefficients').print_rows(num_rows=len(all_features) + 1)

predictions = best_model.predict(testing)
RSS = sum([(predictions[i] - testing[i]['price']) ** 2 for i in range(len(predictions))])
print RSS

best_model['coefficients']['value'].nnz()

max_nonzeros = 7

l1_penalty_values = np.logspace(8, 10, num=20)

info = []
for l1_penalty in l1_penalty_values:
    model = graphlab.linear_regression.create(training, target='price', features=all_features,
                                              validation_set=None, 
                                              l2_penalty=0, l1_penalty=l1_penalty, verbose=False)
    nnz = model['coefficients']['value'].nnz()
    info.append((l1_penalty, nnz))

for x in enumerate(info):
    print x

l1_penalty_min = l1_penalty_values[14]
l1_penalty_max = l1_penalty_values[15]
print l1_penalty_min, l1_penalty_max

l1_penalty_values = np.linspace(l1_penalty_min,l1_penalty_max,20)
print l1_penalty_values

sort_table = []

for l1_penalty in l1_penalty_values:
    model = graphlab.linear_regression.create(training, target='price', features=all_features,
                                              validation_set=None, 
                                              l2_penalty=0, l1_penalty=l1_penalty, verbose=False)
    nnz = model['coefficients']['value'].nnz()
    if not nnz == max_nonzeros:
        continue
    predictions = model.predict(validation)
    RSS = sum([(predictions[i] - validation[i]['price']) ** 2 for i in range(len(predictions))])
    print l1_penalty, RSS
    sort_table.append((RSS, l1_penalty))
    
print sorted(sort_table)[0]

best_model = graphlab.linear_regression.create(training, target='price', features=all_features,
                                              validation_set=None, 
                                              l2_penalty=0, l1_penalty=3320073020.20013, verbose=False)
best_model.get('coefficients').print_rows(num_rows=len(all_features) + 1)

