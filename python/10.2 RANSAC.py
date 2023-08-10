import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
df.columns= ['CRIM', 'ZN', 'INDUS', 'CHAS', 
             'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
             'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
X = df[['RM']].values
X = X.reshape(X.shape[0], -1)
y = df['MEDV'].values
y = y.reshape(y.shape[0], -1)

from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

ransac = RANSACRegressor(LinearRegression(),
               max_trials=100,
               min_samples=50,
               residual_metric=lambda x: np.sum(np.abs(x), axis=1), 
               residual_threshold=5.0,
               random_state=0)

ransac.fit(X,y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_X = np.arange(X.min()-1, X.max()+1, 1)
line_y = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask], color='blue', marker='o', label='inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], color='green', marker='s', label='outliers')
plt.plot(line_X, line_y, color='red')
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc='upper left')
plt.show()

print('Ransac Slope: %.3f' % ransac.estimator_.coef_[0])
print('Ransac Intercept: %.3f' % ransac.estimator_.intercept_)

