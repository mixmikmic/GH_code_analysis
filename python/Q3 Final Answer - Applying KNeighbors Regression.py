import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.path as mplPath
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
get_ipython().magic('matplotlib inline')

def scatter_all(x,y):
    plt.scatter(x,y,color=['red','blue'])
    plt.title('Predicted vs Real Tips')
    plt.xlabel('Predicted Tips')
    plt.ylabel('Real Tips')
    plt.show()

def scatter_less(x,y):
    plt.scatter(x[:2000],y[:2000],color=['red','blue'])
    plt.xlim(-1,40)
    plt.ylim(-1, 40)
    plt.title('Predicted vs Real Tips')
    plt.xlabel('Predicted Tips')
    plt.ylabel('Real Tips')
    plt.show()

def scatter_even_less(x,y):
    plt.scatter(x[:500],y[:500],color=['red','blue'])
    plt.xlim(-1,20)
    plt.ylim(-1, 20)
    plt.title('Predicted vs Real Tips')
    plt.xlabel('Predicted Tips')
    plt.ylabel('Real Tips')
    plt.show()

def knearestRegressor(df, day):
    
    # Break-down by days of the week
    print("Looking at ",day)
    df = df.loc[(df['weekday'] == day)]
    
    # For the future, not sure if it's still running sometimes...
    # Also good to know if we're getting the right data and its shape
    print(df.shape)
    print(df.head())
    
    # Create the training and testing sets
    train = df.sample(frac=0.8, random_state=1)
    test = df.loc[~df.index.isin(train.index)]
    
    # Create the X and Y training and testing sets 
    Xtrain = train[['weekday','hour','pickup','trip_distance','total_amount','time_spent']]
    ytrain = train[['tip_amount']]
    Xtest = test[['weekday','hour','pickup','trip_distance','total_amount','time_spent']]
    ytest = test[['tip_amount']]
    
    # Prepare the matrix for X train and X test
    Xtrain = Xtrain.join(pd.get_dummies(Xtrain['hour']))
    Xtrain = Xtrain.drop(['hour','weekday','pickup'], axis=1)
    Xtest = Xtest.join(pd.get_dummies(Xtest['hour']))
    Xtest = Xtest.drop(['hour','weekday','pickup'], axis=1)
    
    # Perform cross-validation (average 30 minutes for each set)
    # Cross-validation on the training data using the KNeighborsRegressor Classifier
    cross_validation = cross_val_score(KNeighborsRegressor(),Xtrain, ytrain,cv=5)
    
    # Get the accuracy of our training set
    print("Accuracy for ",day ," : %0.4f (+/- %0.4f)" % (cross_validation.mean(), cross_validation.std() * 2))
    
    # Peform KNeighborsRegressor
    clf = KNeighborsRegressor().fit(Xtrain, ytrain)
    score = clf.score(Xtest, ytest)
    print("Score for fold: %.4f" % (score))
    
    # Find out our error MSE and RMSE
    mse = mean_squared_error(clf.predict(Xtest),ytest)
    print("MSE = ",mse)
    print("RMSE = ",np.sqrt(mse))
    
    # Get the predicted data
    prediction_data = clf.predict(Xtest)
    
    # Get the real data
    tip_amt = ytest.tip_amount
    real_data = []
    for tips in tip_amt:
        real_data.append(tips)
        
    # Plot some graphs
    # This will help with the analysis by looking at smaller chunks at a time
    scatter_all(prediction_data,real_data)
    scatter_less(prediction_data,real_data)
    scatter_even_less(prediction_data,real_data)
    return 

df=pd.read_csv("datasets/clean-january-2013.csv") # tippers and non-tippers together

dayweek = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

for day in dayweek:
    knearestRegressor(df,day)



