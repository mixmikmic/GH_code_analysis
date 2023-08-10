import pandas as pd
import numpy as np


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'

# No header on this dataset, so defining columns headings from the documentation.
cols = ['symboling','normalized_losses','make','fuel_type','aspiration','doors','body_style','drive_wheels',
       'engine_location','wheel_base','length','width','height','curb_weight','engine_type','cylinders','engine_size',
       'fuel_system','bore','stroke','compression_ratio','horsepower','peak_rpm','city_mpg','highway_mpg','price']

cars = pd.read_csv(url,header=None,names=cols)

cars.head()

# Replacing all '?' symbols with NaN
cars.replace(to_replace='?',value=np.nan,inplace=True)

cars.dtypes

# Simple transform to float for columns with NaN
num_cols = ['normalized_losses','bore','stroke','horsepower','peak_rpm','price']
#cars[num_cols].isnull().sum()

cars[num_cols] = cars[num_cols].astype(float)

# Dictionary to map string versions of numbers to integers (for doors and cylinders)
str_nums = {'two':2,'three':3,'four':4,'five':5,'six':6,'eight':8,'twelve':12}

cars['doors'] = cars['doors'].map(str_nums)
cars['cylinders'] = cars['cylinders'].map(str_nums)

cars.dtypes

# Looking better. Now to deal with null values.
cars.isnull().sum()

# Vast majority of nulls are in the normalized_losses column, so that'll be the key one to deal with.
# Our dataset is small enough as it is, so I'm not keen to drop those rows. Instead I'll replace nulls
# with the average for the column.

cars['normalized_losses'] = cars['normalized_losses'].fillna(cars['normalized_losses'].mean())

# We'll be predicting price, so it'll be vital to have an accurate price value in our test set. For that reason, we'll drop
# rows with a null price.
cars = cars[cars['price'].isnull() == False]

# There is a complete overlap between null bore and null stroke rows, so we'll drop those too.
cars = cars[cars['bore'].isnull() == False]

# Similarly we'll drop the horsepower and peak_rpm nulls as they overlap
cars = cars[cars['horsepower'].isnull() == False]

# Both the null values in the doors column are on rows with the 'sedan' body type. Sedans are normally 4 door, so we'll
# use that value.
cars['doors'] = cars['doors'].fillna(4)

# Now we will normalize the numeric columns, so each has equal weight in the distance function

# Making a list of the numeric columns to normalize
numeric_cols = cars.select_dtypes(exclude=['object']).columns.tolist()

# Normalization equation: (value - value_mean) / value_standard_deviation 
normalized_cars = (cars[numeric_cols] - cars[numeric_cols].mean())/cars[numeric_cols].std()
normalized_cars['price'] = cars['price']
normalized_cars.head()

# We ought to randomize the order of the dataset first, so we can just cut the set in a straighforward manner later
# while still avoiding any bias based on the order of the dataset.

np.random.seed(1)
shuf_index = np.random.permutation(normalized_cars.index)
normalized_cars = normalized_cars.loc[shuf_index]

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import math

# Function to create our model.
def knn_train_test(k,df,train_col,target_col='price'):
    rows = df.shape[0]
    train_set = df[0:math.floor(rows * 0.75)]
    test_set = df[math.floor(rows * 0.75):]
    
    knn = KNeighborsRegressor(k,algorithm='auto')
    knn.fit(train_set[[train_col]],train_set[target_col])
    predictions = knn.predict(test_set[[train_col]])
    rmse = mean_squared_error(test_set[target_col],predictions)**(1/2)
    return rmse

# We will call the model function on each column in the normalized dataset individually.
cols = normalized_cars.columns.tolist()
cols.remove('price')

# We'll also iterate over a range of k values.
kvals = [1,3,5,7,9]

# List to hold results
rmses = []
for c in cols:
    for ks in kvals:
        rmse = knn_train_test(ks,normalized_cars,c)
        # Results are a tuple with column name, k value and root mean squared value
        rmses.append((c,ks,rmse))

# Converting the results to a dataframe for ease of sorting/filtering/aggregating.
labels = ['column_name','k_value','rmse']
results = pd.DataFrame(rmses,columns=labels)

# We'll take a look at the top six columns, ranked by minimum rmse over all k values
top_6 = results[['column_name','rmse']].groupby('column_name').agg(min).sort_values('rmse').iloc[0:6]
top_6_names = top_6.index.tolist()
top_6

# Visualizing the top 6 columns over the range of k values.

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

fig = plt.figure(figsize=(18, 7))

for i,sp in enumerate(top_6_names):
    ax = fig.add_subplot(2,3,i+1)
    ax.plot(results[results['column_name']==sp]['k_value'], results[results['column_name']==sp]['rmse'], linewidth=3)
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(0, 9)
    ax.set_ylim(2000,6000)
    ax.set_title(sp)
    ax.tick_params(bottom="off", top="off", left="off", right="off")

plt.show()

# Updating the model function so it accepts list parameters for the training columns.

def multi_knn_tt(k,df,train_cols,target_col='price'):
    rows = df.shape[0]
    train_set = df[0:math.floor(rows * 0.75)]
    test_set = df[math.floor(rows * 0.75):]
    
    knn = KNeighborsRegressor(k,algorithm='auto')
    knn.fit(train_set[train_cols],train_set[target_col])
    predictions = knn.predict(test_set[train_cols])
    rmse = mean_squared_error(test_set[target_col],predictions)**(1/2)
    return rmse

# We'll try modelling using combinations of the top performing univariate columns.
training_column_lists = [top_6_names[0:2],top_6_names[0:3],top_6_names[0:4],top_6_names[0:5],top_6_names]
titles = ['Best 2','Best_3','Best 4','Best 5','Best 6']

# New set of kvals to test over
kvals = [x for x in range(1,25)]

# We'll append results to a list
multi_rmses = []
for i,c in enumerate(training_column_lists):
    for ks in kvals:
        rmse = multi_knn_tt(ks,normalized_cars,c)
        # Results are a tuple with title, k value and root mean squared value
        multi_rmses.append((titles[i],ks,rmse))

# Converting the results to a dataframe for ease of sorting/filtering/aggregating.
labels = ['title','k_value','rmse']
multi_results = pd.DataFrame(multi_rmses,columns=labels)

# Display the minimum rmse for each group of features
multi_results[['title','rmse']].groupby('title').agg(min).sort_values('rmse').iloc[0:6]

fig = plt.figure(figsize=(18, 7))

for i,sp in enumerate(titles):
    ax = fig.add_subplot(2,3,i+1)
    ax.plot(multi_results[multi_results['title']==sp]['k_value'], multi_results[multi_results['title']==sp]['rmse'], linewidth=3)
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(0, 25)
    ax.set_ylim(0,5000)
    ax.set_title(sp)
    ax.tick_params(bottom="off", top="off", left="off", right="off")

plt.show()

from sklearn.model_selection import cross_val_score

# We'll run cross validation using 10 folds, over the same feature lists from before
num_folds = 10

results = []

for i,c in enumerate(training_column_lists):
    for ks in kvals:
        # Instantiate the model with the current k value
        model = KNeighborsRegressor(ks)
        
        # Function calls the model and runs over ten folds, returning an array of negative mean square error values.
        mses = cross_val_score(model,normalized_cars[c],normalized_cars['price'],
                               scoring='neg_mean_squared_error',cv=num_folds)
        
        # We need the absolute value of the negative mean square error (returned by the cross val score function)
        root_mses = [abs(m)**(1/2) for m in mses]
        avg_rmse = np.mean(root_mses)
        
        # Logging the results
        results.append([titles[i],ks,avg_rmse])
        
# Converting the results to a dataframe for ease of sorting/filtering/aggregating.
labels = ['title','k_value','average_rmse']
cross_val_results = pd.DataFrame(results,columns=labels)

cross_val_results[['title','average_rmse']].groupby('title').agg(min).sort_values('average_rmse').iloc[0:6]

fig = plt.figure(figsize=(18, 7))

for i,sp in enumerate(titles):
    ax = fig.add_subplot(2,3,i+1)
    ax.plot(multi_results[cross_val_results['title']==sp]['k_value'], 
            cross_val_results[cross_val_results['title']==sp]['average_rmse'], linewidth=3)
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(0, 25)
    ax.set_ylim(0,5000)
    ax.set_title(sp)
    ax.tick_params(bottom="off", top="off", left="off", right="off")

plt.show()

