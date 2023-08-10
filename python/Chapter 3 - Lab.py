import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.datasets import load_boston

boston_raw = load_boston()
boston = pd.DataFrame(data=boston_raw['data'], columns=boston_raw['feature_names'])
boston['MEDV'] = boston_raw['target']
boston.head()

boston.columns

boston_ols = smf.ols('MEDV ~ LSTAT', boston).fit()
boston_ols.summary()

print('Parameters: \n{}\n'.format(boston_ols.params))

boston_conf_int = boston_ols.conf_int(alpha=0.05)
boston_conf_int.columns=['5%', '95%']
print('Confidence Intervals:\n{}'.format(boston_conf_int))

boston_ols.predict(pd.DataFrame({'LSTAT': [5, 10, 15]}))

sns.regplot(x=boston['LSTAT'], y=boston['MEDV'], ci=False, 
            line_kws={'color': 'red'}, marker='+')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(boston_ols.predict(), boston_ols.resid)
ax2.scatter(boston_ols.predict(), boston_ols.outlier_test()['student_resid'])

import statsmodels.api as sm
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ip = sm.graphics.influence_plot(boston_ols, ax=ax)

boston_ols = smf.ols('MEDV ~ LSTAT + AGE', boston).fit()
boston_ols.summary()

#there is no analogy to 'MEDV ~ .' in statsmodels
#but it's easy to create the string of all column names excluding MEDV
#' + '.join(boston.columns[:-1])
boston_ols = smf.ols('MEDV ~ {}'.format(' + '.join(boston.columns[:-1])), boston).fit()
boston_ols.summary()

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

#need to scale the variables
boston_scaled = boston.copy()
for c in boston_scaled.columns[:-1]:
    boston_scaled[c] = boston_scaled[c] - boston_scaled[c].mean()
X = boston_scaled[boston_scaled.columns[:-1]].values
vars = boston_scaled.columns[:-1]
for i in range(X.shape[1]):
    print('VIF {}: {:.4f}'.format(vars[i], vif(X, i)))

#need to take out 'AGE'
boston_ols = smf.ols('MEDV ~ {}'.format(' + '.join(boston.columns[:-1])), boston).fit()
boston_ols.summary()

boston_ols = smf.ols('MEDV ~ LSTAT * AGE', boston).fit()
boston_ols.summary()

boston_ols = smf.ols('MEDV ~ LSTAT + I(LSTAT**2)', boston).fit()
boston_ols.summary()

boston_ols1 = smf.ols('MEDV ~ LSTAT', boston).fit()
boston_ols2 = smf.ols('MEDV ~ LSTAT + I(LSTAT**2)', boston).fit()
sm.stats.anova_lm(boston_ols1, boston_ols2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(boston_ols2.predict(), boston_ols2.resid)
ax2.scatter(boston_ols2.predict(), boston_ols2.outlier_test()['student_resid'])

#this is a replicate of the poly() function in R
def poly(x, p):
    x = np.array(x)
    X = np.transpose(np.vstack((x**k for k in range(p+1))))
    return np.linalg.qr(X)[0][:,1:]

boston_ols = smf.ols('MEDV ~ poly(LSTAT, 5)', boston).fit()
boston_ols.summary()

boston_ols = smf.ols('MEDV ~ np.log(RM)', boston).fit()
boston_ols.summary()

carseats = pd.read_csv('../../data/Carseats.csv', index_col=0)
carseats.head()

carseat_ols = smf.ols('Sales ~ {} + Income:Advertising + Price:Age'                       .format(' + '.join(carseats.columns[1:])), carseats)                       .fit()
carseat_ols.summary()

from patsy.contrasts import Treatment

levels = carseats['ShelveLoc'].unique().tolist()
contrast = Treatment(reference=0).code_without_intercept(levels)
for i in range(len(levels)):
    print(levels[i], contrast.matrix[i, :])

#this is dumb
def load_libraries():
    import pandas
    import numpy 
    print('The libraries have been loaded')

load_libraries()



