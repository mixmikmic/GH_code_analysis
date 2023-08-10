import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
boston = load_boston()

df =  pd.DataFrame( data=boston.data )
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
df['MEDV'] = boston.target
df.head()

X = df[['RM']].values
y = df[['MEDV']].values # Add more [] to avoid getting WARNING in standardization

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='blue')
    plt.plot(X, model.predict(X), color='red')
    return

from sklearn.linear_model import LinearRegression

slr = LinearRegression()
slr.fit(X, y)
print 'Slop: %.3f'% slr.coef_[0]
print 'Intercept: %.3f'% slr.intercept_

plt.figure(figsize=(8,6))
lin_regplot(X, y, slr)
plt.xlabel('Average number of room [RM] (Standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (Standardized)' )
plt.show()

# The input 5 is the same with hand-made results.
print slr.predict([[5.0]])[0,0]

Xb = np.hstack((np.ones((X.shape[0], 1)), X))
w = np.zeros(X.shape[0])
z = np.linalg.inv(np.dot(Xb.T, Xb))
w = np.dot(z, np.dot(Xb.T, y))
print 'Slope: %.3f' % w[1]
print 'Intercept: %.3f' % w[0]

from sklearn.linear_model import RANSACRegressor

ransac = RANSACRegressor( LinearRegression(), 
                          max_trials=100, min_samples=50, 
                          #residual_matric=lambda dy: np.sum(np.abs(dy), axis=1),
                          residual_threshold = 5.0,
                          random_state=0 )
ransac.fit(X, y)

inlier_mask = ransac.inlier_mask_
outier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict( line_X[:, np.newaxis] )

plt.figure(figsize=(8,6))
plt.scatter( X[inlier_mask], y[inlier_mask], c='blue', marker='o', label='Inliner')
plt.scatter( X[outier_mask], y[outier_mask], c='lightgreen', marker='s', label='Outier')
plt.plot( line_X, line_y_ransac, color='red')
plt.xlabel('Average number of room [RM]')
plt.ylabel('Price in $1000\'s [MEDV]' )
plt.show()

from sklearn.model_selection import train_test_split

# Use all variables
X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_predict = slr.predict(X_train)
y_test_predict = slr.predict(X_test)

plt.figure(figsize=(8,6))
plt.scatter(y_train_predict, y_train_predict - y_train, c='blue', marker='o', label='Training data')
plt.scatter(y_test_predict, y_test_predict - y_test, c='lightgreen', marker='o', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residual')
plt.legend(loc='upper left')
plt.hlines( y=0, xmin=-10, xmax=50, lw=2, color='red' )
plt.xlim([-10, 50])
plt.show()

from sklearn.metrics import mean_squared_error, r2_score
print 'Train MSE: %.3f, R2: %.3f'%( mean_squared_error(y_train, y_train_predict), r2_score(y_train, y_train_predict) )
print 'MSE test:  %.3f, R2: %.3f' %( mean_squared_error(y_test,  y_test_predict),  r2_score(y_test,  y_test_predict) )

