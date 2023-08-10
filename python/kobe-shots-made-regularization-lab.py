import numpy as np
import pandas as pd
import patsy

from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import cross_val_score, KFold

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

get_ipython().magic("config InlineBackend.figure_format = 'retina'")
get_ipython().magic('matplotlib inline')

kobe = pd.read_csv('./datasets/kobe_superwide_games.csv')

# A:
kobe.shape

kobe.head(2)

kobe[kobe.columns[:30]].describe()

# Too many columns, regularization able to filter out 'bad' or unnecessary features to avoid overfitting

# A:
# Normalization is necessary because the betas of variables on different scales are biased.
# To apply a constant 'penalty' on these variables will make no sense as it is not fair.
# Thus when variables are on different scales, regularization is unable to filter out important/unimportant features
# accurately

# Standardization is necessary for regularized regression because the beta
# values for each predictor variable must be on the same scale. If betas
# are different sizes just because of the scale of predictor variables
# the regularization term can't determine which betas are more/less 
# important based on their size.

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

Y = kobe[['SHOTS_MADE']]
X = kobe.drop(labels='SHOTS_MADE', axis=1)

Xs = scaler.fit_transform(X)
Xs_df = pd.DataFrame(Xs, columns=X.columns)
Xs_df.describe()

lr = LinearRegression()
kf = KFold(n_splits=10)
scores = cross_val_score(lr, Xs, Y.values.ravel(), cv=10)
print(scores)
print(np.mean(scores))

# A:
ridge_alphas = np.logspace(0, 5, 200)

optimal_ridge = RidgeCV(alphas=ridge_alphas, cv=10)
optimal_ridge.fit(Xs, Y.values.ravel())

print optimal_ridge.alpha_

# A:
ridge = Ridge(alpha=optimal_ridge.alpha_)
ridge_scores = cross_val_score(ridge, Xs, Y.values.ravel(), cv=10)

print ridge_scores
print np.mean(ridge_scores)

# Much better score than plain linear regression
# Ridge manages the betas of non-independent variables, reducing their effects on the regression model,
# in turn preventing overfitting and allowing model to predict better for out-of-sample data.

# A:
optimal_lasso = LassoCV(n_alphas=500, cv=10, verbose=1)
optimal_lasso.fit(Xs, Y.values.ravel())

print optimal_lasso.alpha_

# A:
lasso = Lasso(alpha=optimal_lasso.alpha_)
lasso_scores = cross_val_score(lasso, Xs, Y.values.ravel(), cv=10)

print lasso_scores
print np.mean(lasso_scores)

# Lasso performs slightly better than ridge
# Instead of managing redundant or correlated variables, Lasso removes them completely by reducing their betas to 0,
# removing the effect of these variables on the model completely
# This leaves only the necessary features behind, which might explain the slightly better performance

# A:
lasso.fit(Xs, Y.values.ravel())

len(lasso.coef_)

lasso_coefs = pd.DataFrame({'variable':X.columns,
                            'coef':lasso.coef_,
                            'abs_coef':np.abs(lasso.coef_)})

lasso_coefs.sort_values('abs_coef', inplace=True, ascending=False)

lasso_coefs.head(20)

print 'Percent variables zeroed out:', np.sum((lasso.coef_ == 0))/float(X.shape[0])

# 32% 

# A:
l1_ratios = np.linspace(0.01, 1.0, 25)
optimal_enet = ElasticNetCV(l1_ratio=l1_ratios, n_alphas=100, cv=10,
                            verbose=1)
optimal_enet.fit(Xs, Y.values.ravel())

print optimal_enet.alpha_
print optimal_enet.l1_ratio_

# A:
elasticnet = ElasticNet(alpha=optimal_enet.alpha_, l1_ratio=optimal_enet.l1_ratio_)
elasticnet_scores = cross_val_score(elasticnet, Xs, Y.values.ravel(), cv=10)

print(elasticnet_scores)
print(np.mean(elasticnet_scores))

# performs almost the same as Lasso
# Expected since l1 ratio approaching full lasso, 0.95875

# A: Maybe a jointplot?
ridge.fit(Xs, Y.values.ravel())
lasso.fit(Xs, Y.values.ravel())

ridge_pred = ridge.predict(Xs)
lasso_pred = lasso.predict(Xs)

ridge_resid = Y.values.ravel() - ridge_pred
lasso_resid = Y.values.ravel() - lasso_pred

sns.jointplot(ridge_resid, lasso_resid)



