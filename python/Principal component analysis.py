import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Import the data we want to model
start = '2014-01-01'
end = '2015-01-01'
r1 = get_pricing('MSFT', fields='price', start_date=start, end_date=end).pct_change()[1:]
r2 = get_pricing('SPY', fields='price', start_date=start, end_date=end).pct_change()[1:]

# Run the PCA
pca = PCA() # Create an estimator object
pca.fit(np.vstack((r1,r2)).T)
components = pca.components_
evr = pca.explained_variance_ratio_
print 'PCA components:\n', components
print 'Fraction of variance explained by each component:', evr

# Plot the data
plt.scatter(r1,r2,alpha=0.5)

# Plot the component vectors returned by PCA
xs = np.linspace(r1.min(), r1.max(), 100)
plt.plot(xs*components[0,0]*evr[0], xs*components[0,1]*evr[0], 'r')
plt.plot(xs*components[1,0]*evr[1], xs*components[1,1]*evr[1], 'g')

# Set 1:1 aspect ratio
plt.axes().set_aspect('equal', 'datalim')

# Import a larger-dimension dataset
assets = ['SPY', 'XLE', 'XLY', 'XLP', 'XLI', 'XLU', 'XLK', 'XBI', 'XLB', 'XLF', 'GLD']
returns = get_pricing(assets, fields='price', start_date=start, end_date=end).pct_change()[1:]
print 'Dimension of data:', len(assets)

# Get principal components until 90% of variance is explained
pca_pv = PCA(0.9)
pca_pv.fit(returns)
components_pv = pca_pv.components_
evr_pv = pca_pv.explained_variance_ratio_
print '\nFraction of variance explained by each component up to 0.9 total:', evr_pv

# Get principal components, number of components determined by MLE
pca_mle = PCA('mle')
pca_mle.fit(returns)
evr_mle = pca_mle.explained_variance_ratio_
print '\nNumber of principal components using MLE:', len(evr_mle)
print 'Fraction of variance explained by each component using MLE:', evr_mle

r1_s = r1/r1.std()
r2_s = r2/r2.std()

pca.fit(np.vstack((r1_s,r2_s)).T)
components_s = pca.components_
evr_s = pca.explained_variance_ratio_
print 'PCA components:\n', components_s
print 'Fraction of variance explained by each component:', evr_s

# Plot the data
plt.scatter(r1_s,r2_s,alpha=0.5)

# Plot the component vectors returned by PCA
xs = np.linspace(r1_s.min(), r1_s.max(), 100)
plt.plot(xs*components_s[0,0]*evr_s[0], xs*components_s[0,1]*evr_s[0], 'r')
plt.plot(xs*components_s[1,0]*evr_s[1], xs*components_s[1,1]*evr_s[1], 'g')

# Set 1:1 aspect ratio
plt.axes().set_aspect('equal', 'datalim')

import statsmodels.api as sm
from statsmodels import regression

# Compute returns on factor i, which are returns on portfolio with weights components_pv[i], for all i
factor_returns = np.array([(components_pv[i]*returns).T.sum()
                           for i in range(len(components_pv))])

# Regress first asset against the factors
mlr = regression.linear_model.OLS(returns.T.iloc[0], sm.add_constant(factor_returns.T)).fit()
print 'Regression coefficients for %s\n' % assets[0], mlr.params

