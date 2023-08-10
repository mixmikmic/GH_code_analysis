import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

get_ipython().magic('matplotlib inline')

train = pd.read_csv('./data/Train_UWu5bXk.csv')
test = pd.read_csv('./data/Test_u94Q5KV.csv')

train.info()

test.info()

train.head(3)

sns.regplot(x='Item_MRP', y='Item_Outlet_Sales', data=train);

from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(train[['Item_MRP', 'Item_Visibility', 'Outlet_Establishment_Year']], train.Item_Outlet_Sales, test_size=0.2, random_state=44)

polynomial_features = PolynomialFeatures()
scaler = StandardScaler()
ridge = Ridge()

param_grid_pipeline = {'ridge__alpha': 10. ** np.arange(-2, 2), 'poly__degree': [2, 3, 4]}
linear_pipe = Pipeline([('scaler', scaler), ('poly', polynomial_features), ('ridge', ridge)])

grid = GridSearchCV(linear_pipe, param_grid=param_grid_pipeline, cv=5, scoring='mean_squared_error')

grid.fit(X_train, y_train)

print grid.best_params_

est = grid.best_estimator_
est.fit(X_train, y_train)

predsTrain = est.predict(X_train)
predsTest = est.predict(X_test)

print 'RMSE on training set %f ' %(np.sqrt(mean_squared_error(y_train, predsTrain)))
print 'RMSE on test set %f ' %(np.sqrt(mean_squared_error(y_test, predsTest)))

plt.scatter(predsTrain, predsTrain - y_train, c='b', s=40, alpha=0.5)
plt.scatter(predsTest, predsTest - y_test, c='g', s=40)
plt.hlines(y = 0, xmin=0, xmax = 50)
plt.title('Residual Plot using training (blue) and test (green) data')
plt.ylabel('Residuals');

# fit on the full dataset
est.fit(train[['Item_MRP', 'Item_Visibility', 'Outlet_Establishment_Year']], train.Item_Outlet_Sales)

predictions = est.predict(test[['Item_MRP', 'Item_Visibility', 'Outlet_Establishment_Year']])

submissions = pd.read_csv('./data/SampleSubmission_TmnO39y.csv')

submissions['Item_Outlet_Sales'] = predictions

submissions.to_csv('./submissions/third.csv', index=False)



