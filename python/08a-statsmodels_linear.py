import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

tips = sns.load_dataset('tips')
tips.head()

model = sm.OLS(endog=tips['tip'], exog=tips['total_bill'])

results = model.fit()

results.summary()

# just get the coefficients
results.params

# multiple variable regression
model = sm.OLS(endog=tips['tip'], exog=tips[['total_bill', 'size']])
results = model.fit()
results.summary()

tips.info()

model = smf.ols(formula='tip ~ total_bill + sex + smoker + size',
                data=tips)
results = model.fit()
results.summary()

