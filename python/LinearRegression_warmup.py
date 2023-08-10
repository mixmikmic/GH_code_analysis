import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# this allows plots to appear directly in notebook
get_ipython().magic('matplotlib inline')

data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)

data.head()

data.shape

# visualize the relationship between the features and the response, using scatterplots

fig, axs = plt.subplots(1,3, sharey=True)
data.plot(kind='scatter', x='TV', y='Sales', ax=axs[0], figsize=(16,8))
data.plot(kind='scatter', x='Radio', y='Sales', ax=axs[1])
data.plot(kind='scatter', x='Newspaper', y='Sales', ax=axs[2])

# standard import if you're using "formula notation"
import statsmodels.formula.api as smf

lm = smf.ols(formula='Sales ~ TV', data=data).fit()

lm.params

# lets make a prediction if TV advertising would spend $50,000 
# Statsmodels formula interface expects a datarames
X_new = pd.DataFrame({'TV':[50]})

X_new

lm.predict(X_new)

# create a dataframe with the minimum and maximum values of TV
X_new = pd.DataFrame({'TV':[data.TV.min(), data.TV.max()]})

X_new

preds = lm.predict(X_new)

preds

# first plot the observed data, then plot the least squares line
data.plot(kind='scatter', x='TV', y='Sales')
plt.plot(X_new, preds, c='red', linewidth=2)

# confidence intervals
lm.conf_int()

lm.pvalues

lm.rsquared

# create a fitted model with all three features
lm = smf.ols(formula='Sales ~ TV + Radio + Newspaper', data=data).fit()

lm.params

lm.summary()

# redo above examples with scikit-learn
feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]
y = data.Sales

# usual scikit-learn pattern; import, instantiate, fit

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X,y)

lm.intercept_

lm.coef_

# pair the feature names with the coefficients
zip(feature_cols, lm.coef_)

lm.predict([100, 25, 25])

list(zip(feature_cols, lm.coef_))

# calculate the R-squared
lm.score(X, y)

# set a seed for reproducibility
np.random.seed(12345)

nums = np.random.rand(len(data))
mask_large = nums > 0.5 # random cathegorical data small/large

# initially set Size to small, then change roughly half to be large

data['Size'] = 'small'
data.loc[mask_large,'Size'] = 'large' # apply mask
data.head()

# for scikit-learn, we need to represent all data numerically;

data['IsLarge'] = data.Size.map({'small':0, 'large':1})

data.head()

# redo multiple linear regression and include IsLarge predictor

feature_cols = ['TV', 'Radio', 'Newspaper', 'IsLarge']

X = data[feature_cols]
y = data.Sales

# instantiate, fit
lm = LinearRegression()
lm.fit(X,y)

list(zip(feature_cols, lm.coef_))

# for reproducibilitty
np.random.seed(123456)

# assign roughly one third of observations in each category

nums = np.random.rand(len(data))
mask_suburban = (nums > 0.33) & (nums < 0.66)
mask_urban = (nums > 0.66)
data['Area'] = 'rural'
data.loc[mask_suburban, 'Area'] = 'suburban'
data.loc[mask_urban, 'Area'] = 'urban'
data.head()

# create three dummy variables using get_dummies, then exclude the first dummy column
area_dummies = pd.get_dummies(data.Area, prefix='Area').iloc[:,1:]

area_dummies.head()

data = pd.concat([data, area_dummies], axis=1)
data.head()

feature_cols = feature_cols + ['Area_suburban', 'Area_urban']

feature_cols

X = data[feature_cols]
y = data.Sales

lm = LinearRegression()
lm.fit(X,y)

list(zip(feature_cols, lm.coef_))

lm.predict([100,46,45, 1, 1, 0])



