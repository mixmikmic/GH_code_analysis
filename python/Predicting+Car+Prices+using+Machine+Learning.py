# Importing all the necessary packages 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

col_names = ['symboling','normalized-losses','make','fuel-type','aspiration','num-of-doors','body-style','drive-wheels','engine-location','wheel-base','length','width','height','curb-weight','engine-type','num-of-cylinders','engine-size','fuel-system','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price']
cars = pd.read_csv(r'C:\Users\duttaroym\Documents\Python\imports-85.data', names=col_names)

cars.info()

cars.head()

# Counting missing values
cars['normalized-losses'].value_counts()

cars.drop('normalized-losses',axis = 1, inplace = True)

# Replacing the '?'
cars.replace("?", np.nan, inplace=True)

cars.info()

# Droppping NaN values
cars.dropna(axis = 0, how = 'any', inplace = 'True')

# Correcting the Datatypes of variables
cars[['price','peak-rpm','horsepower']] = cars[['price','peak-rpm','horsepower']].astype(int)
cars[['stroke','bore',]] = cars[['stroke','bore',]].astype(float)

# Importing the necessary SciKit-Learn Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Creating a function to model the data with knn and evaluate using root mean squared error
def knn_train_test(df, col, kval):
    X = df[[col]]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    for k in kval:
        knn = KNeighborsRegressor(k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** (1/2)
        print("RMSE of {} for k = {} : {}".format(col,k, rmse))
        plt.xlim([0,10])
        plt.ylim([0,10000])
        plt.xticks([1,3,5,7,9])
        plt.bar(k,rmse)
                         
    plt.show()      
          
    

features = ['highway-mpg', 'city-mpg', 'peak-rpm', 'horsepower', 'engine-size', 'curb-weight', 
            'width', 'length', 'height', 'wheel-base','bore', 'stroke']

for i in features:
    knn_train_test(cars, i, [1, 3, 5, 7, 9])

two_var = ['engine-size', 'horsepower']
three_var = ['engine-size', 'horsepower','width' ] # For 3 variables
four_var = ['engine-size', 'horsepower', 'width', 'highway-mpg'] # For 4 variables
five_var = ['engine-size', 'horsepower', 'width', 'highway-mpg','city-mpg'] # for 5 variables

# Creating a function to model and evaluate with a specific k value
def knn_train_test1(df, col, k):
    X = df[col]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    knn = KNeighborsRegressor(k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** (1/2)
    print("RMSE of {} for k = {} : {}".format(col,k, rmse))
        

knn_train_test1(cars,two_var, 3)
knn_train_test1(cars, three_var, 3)
knn_train_test1(cars, four_var, 3)
knn_train_test1(cars, five_var, 3)

kval = np.arange(1,26,1) # Listing K-values for hyperparameter optimization

def knn_train_test2(df, col, kval):
    X = df[col]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    for k in kval:
        knn = KNeighborsRegressor(k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** (1/2)
        print("RMSE of {} for k = {} : {}".format(col,k, rmse))
        plt.xlim([0,25])
        plt.ylim([0,7000])
        plt.xticks(kval)
        plt.xlabel("K value")
        plt.ylabel("RMSE")
        plt.title("Error graph for each K value")
        plt.bar(k,rmse)
    plt.show()
    

knn_train_test2(cars, four_var, kval)

