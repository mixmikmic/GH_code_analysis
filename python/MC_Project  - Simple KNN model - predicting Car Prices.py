#import the data
import pandas as pd
pd.options.display.max_columns = 99 #only display 99 (we have over 200 in data set)

cols = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 
        'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
        'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-rate', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
cars = pd.read_csv('imports-85.data',names=cols)
cars.head()

# Select only the columns with continuous values from - https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.names
continuous_values_cols = ['normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'bore', 'stroke', 'compression-rate', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
numeric_cars = cars[continuous_values_cols]

numeric_cars.head()

import numpy as np
#replace '?' with np.nan
numeric_cars=numeric_cars.replace('?',np.nan) 
numeric_cars.info()

# all columns should be numeric. But we see object types. 
# convert all values columns to numeric
# then check for empty rows

numeric_cars = numeric_cars.astype('float')
numeric_cars.isnull().sum()

# because the price column is what we want to predict, lets remove those empty rows
numeric_cars = numeric_cars.dropna(subset=['price'])
numeric_cars.isnull().sum()

# replace remaining missing columns with COLUMN means. Confirm there are no nan left
numeric_cars = numeric_cars.fillna(numeric_cars.mean())
numeric_cars.isnull().sum()

# normalize all columns to range from 0 to 1, except the target column.
# this is important for K-Nearest, as features with inherently larger scales...
# ... can have a greater impact on the Euclidiean formula if NOT normalized.
# I'M PRETTY SURE THIS IS NECESSARY FOR K-NEAREST NEIGHBOR ALGOs

price_col = numeric_cars['price']
numeric_cars = (numeric_cars.max()-numeric_cars)/(numeric_cars.max())
numeric_cars['price'] = price_col
numeric_cars.head()


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

def knn_train_test(train_col, target_col, df):
    knn = KNeighborsRegressor()
    np.random.seed(1)
        
    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_df) / 2)
    
    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_df = rand_df.iloc[0:last_train_row]
    test_df = rand_df.iloc[last_train_row:]
    
    # Fit a KNN model using default k value.
    knn.fit(train_df[[train_col]], train_df[target_col])
    
    # Make predictions using model.
    predicted_labels = knn.predict(test_df[[train_col]])

    # Calculate and return RMSE.
    mse = mean_squared_error(test_df[target_col], predicted_labels)
    rmse = np.sqrt(mse)
    return rmse

rmse_results = {}
train_cols = numeric_cars.columns.drop('price')

# For each column (minus `price`), train a model, return RMSE value
# and add to the dictionary `rmse_results`.
for col in train_cols:
    rmse_val = knn_train_test(col, 'price', numeric_cars)
    rmse_results[col] = rmse_val

# Create a Series object from the dictionary so 
# we can easily view the results, sort, etc
rmse_results_series = pd.Series(rmse_results)
rmse_results_series.sort_values()

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

def knn_train_test(train_col, target_col, df,k):
    knn = KNeighborsRegressor(n_neighbors = k)
    np.random.seed(1)
        
    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_df) / 2)
    
    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_df = rand_df.iloc[0:last_train_row]
    test_df = rand_df.iloc[last_train_row:]
    
    # Fit a KNN model using default k value.
    knn.fit(train_df[[train_col]], train_df[target_col])
    
    # Make predictions using model.
    predicted_labels = knn.predict(test_df[[train_col]])

    # Calculate and return RMSE.
    mse = mean_squared_error(test_df[target_col], predicted_labels)
    rmse = np.sqrt(mse)
    return rmse

# For each column (minus `price`), train a model, return RMSE value
# and add to the dictionary `rmse_results`.

train_cols = numeric_cars.columns.drop('price')
k_rmse_results = {}

k_vals = [1,3,5,7,9]

for col in train_cols:
    col_dict = {}
    for x in k_vals:
        rmse_val = knn_train_test(col, 'price', numeric_cars,x)
        col_dict[x] = rmse_val
    k_rmse_results[col] = col_dict

k_rmse_results

import matplotlib.pyplot as plt
all_labels = []
for k,sub_dicts in k_rmse_results.items():
    x = list(sub_dicts.keys())
    y = list(sub_dicts.values())
    
    plt.plot(x,y)
    plt.xlabel('k (n_neighbors) value')
    plt.ylabel('RMSE value')
    all_labels.append(k)

#include legend and put it right of plot
plt.legend(labels=all_labels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

# which k performed best?
from collections import Counter
min_ks = []
for k, subdict in k_rmse_results.items():
    min_ks.append(min(subdict, key=subdict.get))
    
print(min_ks)
cnt = Counter(min_ks)
print(cnt.most_common(2))

# compute the average RMSE across different 'k' values for each feature.
# the features which result in univariate models with the lowest RMSE vals...
# should be the features you use in your multivariate model

avg_rmses = {}
for k,v in k_rmse_results.items():
    avg_rmses[k] = np.mean(list(v.values()))
    
avg_rmses = pd.Series(avg_rmses).sort_values()
avg_rmses

# make dict of lists of 7 best feature options
num_of_features = [1,2,3,4,5,6,7]
dict_of_lists = {}
for i in num_of_features:
    dict_of_lists[i] = list(avg_rmses.index)[0:i]
dict_of_lists

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

def knn_train_test(train_cols, target_col, df):
    knn = KNeighborsRegressor()
    np.random.seed(1)
        
    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_df) / 2)
    
    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_df = rand_df.iloc[0:last_train_row]
    test_df = rand_df.iloc[last_train_row:]
    
    # Fit a KNN model using default k value.
    knn.fit(train_df[train_cols], train_df[target_col])
    
    # Make predictions using model.
    predicted_labels = knn.predict(test_df[train_cols])

    # Calculate and return RMSE.
    mse = mean_squared_error(test_df[target_col], predicted_labels)
    rmse = np.sqrt(mse)
    return rmse

# use dict_of_lists created above to test model using 7 different sets of features

rmse_results_feats = {}

for k, features_list in dict_of_lists.items():
        rmse_val = knn_train_test(features_list, 'price', numeric_cars)
        rmse_results_feats['%d best features' % k] = rmse_val

rmse_results_feats

# create subset of dictionary with best models (3,4,and 5 features)
best_models_dict = {k: dict_of_lists[k] for k in dict_of_lists.keys() & {3, 4, 5}}
best_models_dict

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

def knn_train_test(train_cols, target_col, df,n_n):
    knn = KNeighborsRegressor(n_n)
    np.random.seed(1)
        
    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_df) / 2)
    
    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_df = rand_df.iloc[0:last_train_row]
    test_df = rand_df.iloc[last_train_row:]
    
    # Fit a KNN model using default k value.
    knn.fit(train_df[train_cols], train_df[target_col])
    
    # Make predictions using model.
    predicted_labels = knn.predict(test_df[train_cols])

    # Calculate and return RMSE.
    mse = mean_squared_error(test_df[target_col], predicted_labels)
    rmse = np.sqrt(mse)
    return rmse

# use best_models_dict created above to test model using 7 different sets of features

rmse_results_final = {}
n_n_list = list(range(1,25))
for k, features_list in best_models_dict.items():
        single_mod = {}
        for n in n_n_list:
            rmse_val = knn_train_test(features_list, 'price', numeric_cars,n)
            single_mod[n] = rmse_val
        rmse_results_final['%d best features' % k] = single_mod

rmse_results_final

import matplotlib.pyplot as plt
all_labels = []
for k,sub_dicts in rmse_results_final.items():
    x = list(sub_dicts.keys())
    y = list(sub_dicts.values())
    
    plt.plot(x,y)
    plt.xlabel('k (n_neighbors) value')
    plt.ylabel('RMSE value')
    all_labels.append(k)

#include legend and put it right of plot
plt.legend(labels=all_labels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

