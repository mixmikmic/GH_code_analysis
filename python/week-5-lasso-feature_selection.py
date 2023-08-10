import graphlab

sales = graphlab.SFrame('kc_house_data.gl/')

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

model_all = graphlab.linear_regression.create(sales, target='price', features=all_features,
                                              validation_set=None, 
                                              l2_penalty=0., l1_penalty=1e10)

model_all['coefficients']['name'][model_all['coefficients']['value']!=0]

(training_and_validation, testing) = sales.random_split(.9,seed=1) # initial train/test split
(training, validation) = training_and_validation.random_split(0.5, seed=1) # split training into train and validate

import numpy as np

i = 0
for l1 in np.logspace(1, 7, num = 13):
    model = graphlab.linear_regression.create(training, target='price', 
                                              features=all_features,
                                              validation_set=None,
                                              l2_penalty=0., l1_penalty=l1,
                                              verbose = False)
    RSS = sum((model.predict(validation) - validation['price'])**2)
    if i == 0:
        model_best = model
        RSS_min = RSS
        i = i + 1
    else:
        if RSS < RSS_min:
            model_best = model
            RSS_min = RSS
print RSS_min
print model_best['coefficients']['value'].nnz()

max_nonzeros = 7

l1_penalty_values = np.logspace(8, 10, num=20)

i = 0
for l1 in np.logspace(8, 10, num = 20):
    model = graphlab.linear_regression.create(training, target='price', 
                                              features=all_features,
                                              validation_set=None,
                                              l2_penalty=0., l1_penalty=l1,
                                              verbose = False)
    RSS = sum((model.predict(validation) - validation['price'])*2)
    i = i+1
    print 'Model %d:' % i
    print model['coefficients']['value'].nnz()

l1_penalty_min = np.logspace(8, 10, num = 20)[-6]
l1_penalty_max = np.logspace(8, 10, num = 20)[-5]

l1_penalty_values = np.linspace(l1_penalty_min,l1_penalty_max,20)

i = 0
for l1 in l1_penalty_values:
    model = graphlab.linear_regression.create(training, target='price', 
                                              features=all_features,
                                              validation_set=None,
                                              l2_penalty=0., l1_penalty=l1,
                                              verbose = False)
    RSS = sum((model.predict(validation) - validation['price'])**2)
    num_nonzero = model['coefficients']['value'].nnz()
    if num_nonzero <= 7:
        if i == 0:
            model_best = model
            RSS_min = RSS
            l1_best = l1
            i = i + 1
        else:
            if RSS < RSS_min:
                model_best = model
                RSS_min = RSS
                l1_best = l1
print RSS_min
print model_best['coefficients']['value'].nnz()
print l1_best

model_best['coefficients']



