import pandas as pd

shd_csv = './datasets/sacramento_real_estate_transactions_Clean.csv'
shd = pd.read_csv(shd_csv)

shd.head(2)

shd['type'].unique()

type_dummy = pd.get_dummies(shd['type'])
type_dummy.head()

type_dummy.drop('Unkown', axis=1, inplace=True)
shd = pd.concat([shd, type_dummy], axis=1)
shd.head(2)

# im going to create a dummy variable for HUGE houses.  
# Those whose square footage is 3 standard deviations away from the mean. 
# - Mean = 1315
# - STD = 853
# - Huge Houses > 3775 sq ft

big = []
for home in shd['sq__ft']:
    if home >= 3775:
        big.append(1)
    else:
        big.append(0)

shd['Huge_homes'] = big

shd['Huge_homes'].value_counts()

# Importing the stats model API
import statsmodels.api as sm

# Setting my X and y for modeling
X = shd[['sq__ft','beds','baths','Huge_homes']]
y = shd['price']

# The Default here is Linear Regression (ordinary least squares regression OLS)
model = sm.OLS(y,X).fit()

y_pred = model.predict(X)

shd['y_pred'] = y_pred
shd['Residuals'] = shd['price'] - shd['y_pred']

import seaborn as sns
get_ipython().magic('matplotlib inline')

sns.lmplot(x='price', y='y_pred', data=shd, hue='Huge_homes')

# Normality:  Do the Residual Errors follow a normal distribution?

# I believe all those properties with 0 values are causing the Y intercept to be higher up
# resulting in a less steep slope, thus creating areas where residual error is higher.
sns.distplot(shd['Residuals'])

# The errors are more or less skewed to the right, but do show approximate normality otherwise.

# Equality of variance.  

# I believe all the observations with 0 sq ft are obscuring the predictive 
# trend so as prices increase error is also going to as well.

sns.lmplot(x='sq__ft', y='Residuals', data=shd)

# Those zero square foot properties are causing some havok and there seems to be trend.
# (assumption is violated)

# When sq__ft, beds, baths, and huge_homes are all 0, the price of the house is estimated to be
# the value of the intercept.

model.summary()

# A: 

