get_ipython().magic('matplotlib inline')
# imports
import pandas as pd
import matplotlib.pyplot as plt

# read data into a DataFrame
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
data.head()

# set a seed for reproducibility
np.random.seed(12345)

# create a Series of booleans in which roughly half are True
nums = np.random.rand(len(data))
mask_large = nums > 0.5

# initially set Size to small, then change roughly half to be large
data['Size'] = 'small'
data.loc[mask_large, 'Size'] = 'large'
data.head()

feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]
y = data.Sales

# follow the usual sklearn pattern: import, instantiate, fit
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X, y)

# print intercept and coefficients
coef = zip(lm.coef_, feature_cols)
print 'model is "sales = {:.2f} + {:.2f}.{} + {:.2f}.{} + {:.2f}.{}"'.format(lm.intercept_, coef[0][0], coef[0][1], coef[1][0], coef[1][1], coef[2][0], coef[2][1])
print 'r2 = ', lm.score(X, y)

X = data[feature_cols[:2]]
y = data.Sales

# follow the usual sklearn pattern: import, instantiate, fit
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X, y)

# print intercept and coefficients
coef = zip(lm.coef_, feature_cols)
print 'model is "sales = {:.2f} + {:.2f}.{} + {:.2f}.{}"'.format(lm.intercept_, coef[0][0], coef[0][1], coef[1][0], coef[1][1])
print 'r2 = ', lm.score(X, y)

# TODO add image and put this code into an appendix at the bottom
from mpl_toolkits.mplot3d import Axes3D


## Create the 3d plot -- skip reading this
# TV/Radio grid for 3d plot
xx1, xx2 = np.meshgrid(np.linspace(X.TV.min(), X.TV.max(), 100), 
                       np.linspace(X.Radio.min(), X.Radio.max(), 100))

# plot the hyperplane by evaluating the parameters on the grid
Z = lm.intercept_ + (lm.coef_[0] * xx1) + (lm.coef_[1] * xx2)

# create matplotlib 3d axes
fig = plt.figure(figsize=(12, 8))
ax = Axes3D(fig, azim=-115, elev=15)

# plot hyperplane
surf = ax.plot_surface(xx1, xx2, Z, cmap=plt.cm.RdBu_r, alpha=0.6, linewidth=0)

# plot data points - points over the HP are white, points below are black
resid = y - lm.predict(X)
ax.scatter(X[resid >= 0].TV, X[resid >= 0].Radio, y[resid >= 0], color='black', alpha=1.0, facecolor='white')
ax.scatter(X[resid < 0].TV, X[resid < 0].Radio, y[resid < 0], color='black', alpha=1.0)

# set axis labels
ax.set_xlabel('TV')
ax.set_ylabel('Radio')
ax.set_zlabel('Sales')

# Z = lm.intercept_ + (lm.coef_[0] * xx1) + (lm.coef_[1]
xx1.shape

# load the boston housing dataset - median house values in the Boston area
df = pd.read_csv('http://vincentarelbundock.github.io/Rdatasets/csv/MASS/Boston.csv')
df.head()

feature_cols = ['crim', 'lstat']
X = df[feature_cols]
y = df.medv

# follow the usual sklearn pattern: import, instantiate, fit
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X, y)

# print intercept and coefficients
for coef, feature in zip(lm.coef_, feature_cols):
    print '{}\t{:.3f}'.format(feature, coef)
    
    
print lm.score(X, y)

feature_cols = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']
X = df[feature_cols]
y = df.medv

# follow the usual sklearn pattern: import, instantiate, fit
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X, y)

# print intercept and coefficients
for coef, feature in zip(lm.coef_, feature_cols):
    print '{}\t{:.3f}'.format(feature, coef)
    
    
print lm.score(X, y)

from sklearn.metrics import r2_score
r2_score(lm.predict(X), y)  

df = pd.read_csv('http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/SAheart.data')
df.head()



