import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

shd_csv = './datasets/sacramento_real_estate_transactions_Clean.csv'

df = pd.read_csv(shd_csv)
df.info()

df = df.drop(labels='Unnamed: 0', axis=1)

df['type'].unique()

# A:
dummy_var = pd.get_dummies(df[['type']])

dummy_var.columns

# A:
dummy_var_drop = dummy_var.drop(labels='type_Unkown',axis=1)

dummy_var_drop.shape

new_df = pd.concat([df,dummy_var_drop], axis=1)
new_df.info()

# A:
import statsmodels.api as sm

new_df.head(2)

print(new_df['street'].unique().size)
print(new_df['city'].unique().size)
print(new_df['zip'].unique().size)
print(new_df['state'].unique().size)
print(new_df['sale_date'].unique().size)

dummy_var = pd.get_dummies(new_df['sale_date'])

# use friday as reference
dummy_var = dummy_var.drop(labels='Fri May 16 00:00:00 EDT 2008', axis=1)

temp_df = new_df.drop(labels=['street','city','zip','state','type','sale_date','latitude','longitude'], axis=1)

temp_df = pd.concat([temp_df, dummy_var], axis=1)

temp_df.head()

temp_df.describe()

### find all houses with beds, baths, sq__ft = 0
print(temp_df[temp_df['beds']==0].index)
print(temp_df[temp_df['baths']==0].shape)
print(temp_df[temp_df['sq__ft']==0].shape)

# lets drop those with 0 beds and baths first
temp_df_drop = temp_df.drop(temp_df[temp_df['beds']==0].index)

# fill up the missing sqft with median values
median_sqft = temp_df_drop['sq__ft'].median()
temp_df_drop.loc[temp_df_drop.sq__ft.values == 0, 'sq__ft'] = median_sqft

import seaborn as sns
import matplotlib.pyplot as plt
import patsy
get_ipython().magic('matplotlib inline')

# EDA
plt.figure(figsize=(15,10))
sns.heatmap(temp_df_drop.corr()**2,annot=True)

# big homes are homes with sq__ft 3 standard deviations away from the median
big_home_cond = temp_df_drop['sq__ft'].median() + 3 * temp_df_drop['sq__ft'].std()
round(big_home_cond)

big = []
for home in temp_df_drop['sq__ft']:
    if home >= 3301:
        big.append(1)
    else:
        big.append(0)

temp_df_drop['Huge_homes'] = big

# get sq_ft, baths, beds as predictors
final_df = temp_df_drop[['beds','baths','sq__ft','price','Huge_homes']]

sns.heatmap(final_df.corr()**2, annot=True)

# build MLR model with huge homes
formula = 'price ~ beds + baths + sq__ft + Huge_homes -1'
y, X = patsy.dmatrices(formula, data=final_df, return_type='dataframe')
# X = final_df[['beds','baths','sq__ft','Huge_homes']]
# y = final_df[['price']]
model = sm.OLS(y,X).fit()
model.summary()

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns



yhat = model.predict(X)

# get predictions from mlr model
final_df['y_pred'] = yhat

# A:
sns.lmplot(x='price', y='y_pred', data=final_df, hue='Huge_homes')

# A:
# LINE
# linearity between predictors and target
# Indepedence of predictors and respective residuals
# Normality of distribution for residuals
# Equality of variances, no relationship between residual and predictors

# get residuals
final_df['residuals'] = final_df['price'] - final_df['y_pred']

# A:
sns.distplot(final_df['residuals'])

sns.lmplot(x='sq__ft', y='residuals', data=final_df)

# A:
# when baths and sqft is 0, the price of home is the intercept

# A:
model.summary()

# A: 
# Model prediction for huge homes has inverted relationship with actual prices

