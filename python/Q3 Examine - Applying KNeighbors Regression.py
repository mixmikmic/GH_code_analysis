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

df=pd.read_csv("datasets/clean-january-2013.csv") # tippers and non-tippers together
#df=pd.read_csv("datasets/cleaner-january-2013.csv") # only those that paid tips

df.head()

df.tail()

df = df.loc[(df['weekday'] == 'Monday')]

# Generate the training set.  Set random_state to be able to replicate results.
train = df.sample(frac=0.8, random_state=1)
# Select anything not in the training set and put it in the testing set.
test = df.loc[~df.index.isin(train.index)]
# Print the shapes of both sets.
print(train.shape)
print(test.shape)

train.head()

Xtrain = train[['weekday','hour','pickup','trip_distance','total_amount','time_spent']]
ytrain = train[['tip_amount']]
Xtest = test[['weekday','hour','pickup','trip_distance','total_amount','time_spent']]
ytest = test[['tip_amount']]
Xtrain.shape, ytrain.shape, Xtest.shape, ytest.shape

Xtrain.head() # training set data we're interested in

ytrain.head() # corresponding tips

## might be useful
pickup_train = Xtrain[['pickup']] # remember the pickup coordinates

Xtrain = Xtrain.join(pd.get_dummies(Xtrain['hour']))
Xtrain = Xtrain.drop(['hour','weekday','pickup'], axis=1)
Xtrain.head()

## might be useful
pickup_test = Xtest[['pickup']] # remember the pickup coordinates

Xtest = Xtest.join(pd.get_dummies(Xtest['hour']))
Xtest = Xtest.drop(['hour','weekday','pickup'], axis=1)
Xtest.head()

Xtrain.shape

get_ipython().run_cell_magic('time', '', 'cross_validation = cross_val_score(KNeighborsRegressor(),Xtrain, ytrain,cv=5)\ncross_validation')

print("Accuracy: %0.2f (+/- %0.2f)" % (cross_validation.mean(), cross_validation.std() * 2))

get_ipython().run_cell_magic('time', '', 'clf = KNeighborsRegressor().fit(Xtrain, ytrain)\nscore = clf.score(Xtest, ytest)\nprint("Score for fold: %.3f" % (score))')

get_ipython().run_cell_magic('time', '', 'mse = mean_squared_error(clf.predict(Xtest),ytest)')

print("MSE = ",mse)
print("RMSE = ",np.sqrt(mse))

get_ipython().run_cell_magic('time', '', 'monday_array = clf.predict(Xtest)')

get_ipython().run_cell_magic('time', '', 'tip_amt = ytest.tip_amount\nreal_tips = []\nfor tips in tip_amt:\n    real_tips.append(tips)\nplt.scatter(monday_array,real_tips)')

get_ipython().run_cell_magic('time', '', "plt.scatter(monday_array,real_tips,color=['red','blue'])\nplt.title('Predicted vs Real Tips')\nplt.xlabel('Predicted Tips')\nplt.ylabel('Real Tips')\nplt.show()")

get_ipython().run_cell_magic('time', '', "plt.scatter(monday_array[:10000],real_tips[:10000],color=['red','blue'])\nplt.xlim(-1,15)\nplt.ylim(-1, 15)\nplt.title('Predicted vs Real Tips (10000 pts)')\nplt.xlabel('Predicted Tips')\nplt.ylabel('Real Tips')\nplt.show()")

get_ipython().run_cell_magic('time', '', "plt.scatter(monday_array[:2000],real_tips[:2000],color=['red','blue'])\nplt.xlim(-1,20)\nplt.ylim(-1, 20)\nplt.title('Predicted vs Real Tips (2000 pts)')\nplt.xlabel('Predicted Tips')\nplt.ylabel('Real Tips')\nplt.show()")

k = np.column_stack((monday_array,real_tips))

print("Predicted vs Real")
print("Show the data")
k[:50]

