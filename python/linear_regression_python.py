import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.datasets import load_boston

boston = load_boston()
#print(boston)
#type(boston)
#boston.feature_names
#print(boston.data.shape)
print(boston.data)

bos = pd.DataFrame(boston.data)
#print(bos.head)
#print(boston.feature_names)
bos.columns = boston.feature_names
#print(bos.head())
#print(boston.target)
#print(bos.describe())
bos['PRICE'] = boston.target
bos.head(3)

X = bos.drop('PRICE', axis = 1)
Y = bos['PRICE']



#Divide the data into training and test set... 
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.33, random_state = 5)

X_train

#Let's quickly look at our data set how it looks like
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

#Step 3 fit the model
lm = LinearRegression()
lm.fit(X_train, Y_train)

#Step 4
Y_test_pred = lm.predict(X_test)

#Step 5 Learn and improve
#Before we learn we need to check how we have fared.. 

df=pd.DataFrame(Y_test_pred,Y_test)
print(df)

mse = mean_squared_error(Y_test, Y_test_pred)
print(mse)

plt.scatter(Y_train_pred, Y_train_pred - Y_train,c='blue',marker='o',label='Training data')
plt.scatter(Y_test_pred, Y_test_pred - Y_test,c='red',marker='s',label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc= 'upper left')
plt.hlines(y=0,xmin=0,xmax=50)
plt.plot()
plt.show()



