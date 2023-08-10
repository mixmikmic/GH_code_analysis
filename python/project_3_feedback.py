import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import patsy
import statsmodels.api as sm
import scipy.stats as ss
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.linear_model import ElasticNet, ElasticNetCV, SGDRegressor, SGDClassifier, LinearRegression
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn import metrics

sns.set_style('whitegrid')

get_ipython().magic("config InlineBackend.figure_format = 'retina'")
get_ipython().magic('matplotlib inline')

# Load the data
df = pd.read_csv('./housing.csv')
house = df

# Make column headers small caps
cols = [x.lower() for x in house.columns]
house.columns = cols

# Check for null values:
house.isnull().sum().sort_values(ascending=False)

# Drop PoolQC, PoolArea, MiscFeature, Alley, Fence
house.drop(labels=['poolqc','poolarea','miscfeature','miscval','alley','fence'], axis=1, inplace=True)

# Drop Id Column
house.drop(labels='id', axis=1, inplace=True)

# Dealing with Null for Fire Place Quality
house.fireplacequ.value_counts()

# Replace Null with NA means no fire place
# fill fire place quality nan with NA
house.fireplacequ.fillna('NA', inplace=True)

def null_weighted_imputation(df_series):
    total_abv_0 = df_series.where(df_series > 0).value_counts().sum()
    weight_df = (df_series.where(df_series > 0).value_counts() / total_abv_0).to_frame()
    values = weight_df.index.tolist()
    weights = weight_df.values.ravel()
    df_series = df_series.fillna(np.random.choice(values, p=weights))
    return df_series

# Dealing with null in Lot Frontage
# Impute with weighted random selection from distribution
series = null_weighted_imputation(house.lotfrontage)
house.lotfrontage = series

# Dealing with null for Garage columns
house[house.garagecond.isnull()].loc[:,['garagecond','garagetype','garageyrblt','garagefinish','garagequal']]

# All nan values occur in same row
# Fill them with NA meaning no garage
# Garage year fill with median
house.garageyrblt.fillna(house.garageyrblt.median(), inplace=True)
house.garagetype.fillna('NA', inplace=True)
house.garagefinish.fillna('NA', inplace=True)
house.garagequal.fillna('NA', inplace=True)
house.garagecond.fillna('NA', inplace=True)

# Dealing with nan values in basement columns
house[house.bsmtfintype2.isnull()].loc[:,['bsmtexposure','bsmtfintype2','bsmtfintype1','bsmtcond','bsmtqual']]

# Nan values all in same row
# There is 1 nan in bsmt exposure & bsmtfintype2 that must be replaced by mode
print house.bsmtexposure.value_counts().sort_values(ascending=False)
print house.bsmtfintype2.value_counts().sort_values(ascending=False)

house.loc[948,'bsmtexposure'] = 'No'
house.loc[332,'bsmtfintype2'] = 'Unf'

# The rest can be replaced with none meaning no basement
house.bsmtexposure.fillna('NA', inplace=True)
house.bsmtfintype2.fillna('NA', inplace=True)
house.bsmtfintype1.fillna('NA', inplace=True)
house.bsmtcond.fillna('NA', inplace=True)
house.bsmtqual.fillna('NA', inplace=True)

# fill MasVnrType nan with 'None' & MasVnrArea nan with 0
house[house.masvnrtype.isnull()].loc[:,['masvnrtype','masvnrarea']]

house.masvnrtype.fillna('None', inplace=True)
house.masvnrarea.fillna(0, inplace=True)

# replace nan for electrical with mode
house.electrical.value_counts().sort_values(ascending=False)

house.electrical.fillna('SBrkr', inplace=True)

# All columns have no null

# Now lets do some feature engineering
# Setting dtypes, Combinations, time and ordinal columns

# drop mssubclass, its a combination of year built,bldgtype, housestyle
house.drop(labels='mssubclass', axis=1, inplace=True)

# condition 1 & 2 can be combined
house['condition'] = house.condition1 + house.condition2
house.drop(labels=['condition1','condition2'], axis=1, inplace=True)
# exterior1st and exterior2nd can be combined
house['exterior'] = house.exterior1st + house.exterior2nd
house.drop(labels=['exterior1st','exterior2nd'], axis=1, inplace=True)
# combine bsmtfinsf1 / 2 then divide by totalbsmtsf to get percent done
house['bsmtfinsf'] = house.bsmtfinsf1 + house.bsmtfinsf2
house.drop(labels=['bsmtfinsf1','bsmtfinsf2'], axis=1, inplace=True)
# (grlivarea - lowqualfinsf) / grlivarea to get percent high quality
house['highqualfinsf_perc'] = (house.grlivarea - house.lowqualfinsf) / house.grlivarea
house.drop(labels=['1stflrsf','2ndflrsf','lowqualfinsf'], axis=1, inplace=True)
# add total rooms above ground & bath rooms above ground & basement to get total rooms in house
# without basement bedrooms
house['totrms_without_bsmtbedrm'] = house.totrmsabvgrd + house.fullbath + house.halfbath + house.bsmtfullbath + house.bsmthalfbath
house.drop(labels=['totrmsabvgrd','fullbath','halfbath','bsmtfullbath','bsmthalfbath','bedroomabvgr','kitchenabvgr'], axis=1, inplace=True)
# combine deck and porch square feet
house['totdeckporchsf'] = house.wooddecksf + house.openporchsf + house.enclosedporch + house['3ssnporch'] + house.screenporch
house.drop(labels=['wooddecksf','openporchsf','enclosedporch','3ssnporch','screenporch'], axis=1, inplace=True)

# transform ordinal features into their numerical equivalents
# the quality / condition columns
house.exterqual = house.exterqual.map(lambda x: 5 if x == 'Ex' else 4 if x == 'Gd' else 3 if x == 'TA' 
                                      else 2 if x == 'Fa' else 1 if x == 'Po' else x)
house.extercond = house.extercond.map(lambda x: 5 if x == 'Ex' else 4 if x == 'Gd' else 3 if x == 'TA' 
                                      else 2 if x == 'Fa' else 1 if x == 'Po' else x)
house.bsmtqual = house.bsmtqual.map(lambda x: 5 if x == 'Ex' else 4 if x == 'Gd' 
                                      else 3 if x == 'TA' else 2 if x == 'Fa' else 1 if x == 'Po' else x)
house.bsmtcond = house.bsmtcond.map(lambda x: 5 if x == 'Ex' else 4 if x == 'Gd' 
                                      else 3 if x == 'TA' else 2 if x == 'Fa' else 1 if x == 'Po' else x)
house.bsmtexposure = house.bsmtexposure.map(lambda x: 4 if x == 'Gd' else 3 if x == 'Av' else 2 if x == 'Mn' 
                                            else 1 if x == 'No' else x)
house.bsmtfintype1 = house.bsmtfintype1.map(lambda x: 6 if x == 'GLQ' else 5 if x == 'ALQ' else 4 if x == 'BLQ' 
                                      else 3 if x == 'Rec' else 2 if x == 'LwQ' else 1 if x == 'Unf' else x)
house.bsmtfintype2 = house.bsmtfintype2.map(lambda x: 6 if x == 'GLQ' else 5 if x == 'ALQ' else 4 if x == 'BLQ' 
                                      else 3 if x == 'Rec' else 2 if x == 'LwQ' else 1 if x == 'Unf' else x)
house.heatingqc = house.heatingqc.map(lambda x: 5 if x == 'Ex' else 4 if x == 'Gd' 
                                      else 3 if x == 'TA' else 2 if x == 'Fa' else 1 if x == 'Po' else x)
house.kitchenqual = house.kitchenqual.map(lambda x: 5 if x == 'Ex' else 4 if x == 'Gd' 
                                      else 3 if x == 'TA' else 2 if x == 'Fa' else 1 if x == 'Po' else x)
house.fireplacequ = house.fireplacequ.map(lambda x: 5 if x == 'Ex' else 4 if x == 'Gd' 
                                      else 3 if x == 'TA' else 2 if x == 'Fa' else 1 if x == 'Po' else x)
house.garagequal = house.garagequal.map(lambda x: 5 if x == 'Ex' else 4 if x == 'Gd' 
                                      else 3 if x == 'TA' else 2 if x == 'Fa' else 1 if x == 'Po' else x)
house.garagecond = house.garagecond.map(lambda x: 5 if x == 'Ex' else 4 if x == 'Gd' 
                                      else 3 if x == 'TA' else 2 if x == 'Fa' else 1 if x == 'Po' else x)

# Transform year columns into age
house['age'] = house.yrsold - house.yearbuilt
house['age_remodel'] = house.yrsold - house.yearremodadd
house.drop(labels=['yearbuilt','yearremodadd'], axis=1, inplace=True)
house['garageage'] = house.yrsold - house.garageyrblt
house.drop(labels='garageyrblt', axis=1, inplace=True)
# Transform mosold into a circular dataset but dont know how
# Month 12 and month 1 are very close not very far...
# Alternative is to create dummies for each month which is what i will do
house.mosold = house.mosold.astype('object')
house.yrsold = house.yrsold.astype('object')

# Add fintype 1 and fintype 2 
house['bsmtfintype'] = house.bsmtfintype1 + house.bsmtfintype2
house.drop(labels=['bsmtfintype1','bsmtfintype2'], axis=1, inplace=True)

# only want residential
house.mszoning.unique()
house = house[house.mszoning != 'C (all)']

house.reset_index(drop=True, inplace=True)
house.info()

# Remove outliers for better prediction
sns.boxplot(house.saleprice)

# Distribution has many outliers
# Transform it via log
plt.figure(figsize=(15,5))
sns.distplot(np.log(house.saleprice))
plt.axvline(np.log(house.saleprice.mean()), color='g', linestyle='dashed', linewidth=2, label='Mean')
plt.axvline(np.log(house.saleprice.median()), color='r', linestyle='dashed', linewidth=2, label='Median')
plt.legend()

# The logged price distribution more normally distributed than original
# We shall use the log sale price as our target
house.saleprice = np.log(house.saleprice)

# Observe the Variations of Sales Price and Types and Style of Dwelling among different Zoning Classifications
fig, ax = plt.subplots(3, 1, sharex='col', sharey='row', figsize=(20,15))

palette_1 = sns.light_palette("purple", reverse=False,  n_colors=len(house.bldgtype.unique()))
palette_2 = sns.light_palette("blue", reverse=False,  n_colors=len(house.housestyle.unique()))
plt.title('Sale Price Distribution for each Sale Class + Dwelling Types / Style ')

sp_75p = house.saleprice.describe()['75%']
sp_med = house.saleprice.median()

sns.boxplot(x=house.mszoning, y=house.saleprice, ax=ax[0])
sns.swarmplot(x=house.mszoning, y=house.saleprice, hue=house.bldgtype, palette=palette_1, ax=ax[1])
sns.swarmplot(x=house.mszoning, y=house.saleprice, hue=house.housestyle, palette=palette_2, ax=ax[2])

x = plt.gca().axes.get_xlim()
ax[0].plot(x, len(x) * [sp_med], 'r--')
ax[0].plot(x, len(x) * [sp_75p], 'g--')
ax[1].plot(x, len(x) * [sp_med], 'r--')
ax[1].plot(x, len(x) * [sp_75p], 'g--')
ax[2].plot(x, len(x) * [sp_med], 'r--')
ax[2].plot(x, len(x) * [sp_75p], 'g--')

# Get conservative estimate of profit potential for each Classification
# Assuming we buy at expected price (median Sale Price) and sell at 75th Percentile
mszoning_stats = house.groupby('mszoning')['saleprice'].describe()
plt.title('IQR of Sale Price for each Sale Class')
plt.ylabel('Sale Price')
(mszoning_stats.loc[:,'75%'] - mszoning_stats.loc[:,'50%']).plot(kind='bar')

# Residential Low Density (RL) housing captures the most number of sales transactions,  
# and the largest profit potential (75th percentile - 50th percentile)
# We can hypothesize that buyers are generally more interested in RL houses
# and are more willing to pay a premium for such houses.
# RL housing market suits our short term goal very well

# They say location affects the prices of real estate
# Lets investigate location features:
# 'Neighborhood','Condition1','Condition2'
fig, ax = plt.subplots(2, 1, sharey='row', figsize=(20,10))
plt.title('Sale Price Distribution for each Neighborhood + Condition')

sns.swarmplot(x=house.neighborhood, y=house.saleprice, hue=house.mszoning, ax=ax[0])
sns.swarmplot(x=house.condition, y=house.saleprice, ax=ax[1])

x = ax[0].get_xlim()
ax[0].plot(x, len(x) * [sp_med], 'r--')
ax[0].plot(x, len(x) * [sp_75p], 'g--')
ax[1].plot(x, len(x) * [sp_med], 'r--')
ax[1].plot(x, len(x) * [sp_75p], 'g--')

# From the chart above, we can find out what are the sale price distribution for each neighborhood and condition

# The red and green line signifies the median and the 75th percentile Sale Price respectively
# For our short term goal we will want to identify neighborhoods with Sale Prices that lie within this range
# The higher the bulk of the distribution the lies within the range, the higher the chance of us executing our
# flipping strategy

# plot Year columns against Sale Price
temp_df = house[['saleprice','age','age_remodel','garageage']]
sns.pairplot(temp_df)

# It seems like the younger the property, the higher the sale price 
# but in actual fact it might be just inflation

# Divide data into continuous and categorical columns
con_cols = [x for x in house.columns if house[x].dtype == 'int64' or house[x].dtype == 'float64']
cat_cols = [x for x in house.columns if x not in con_cols]

# Plot continuous variables + Sale Price
corr = house[con_cols].corr()**2

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 10))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, annot=True, mask=mask, ax=ax)

# test significance for highly correlated values
print ss.pearsonr(house.garagecars, house.garagearea)
print ss.pearsonr(house.grlivarea, house.totrms_without_bsmtbedrm)

# reject null hypothesis, there is significant correlation
house.drop(labels=['garagecars','totrms_without_bsmtbedrm'], axis=1, inplace=True)

# len of categorical features
cat_cols = np.array(cat_cols)
cat_cols.shape = (17,2)

# Plot Categorical features against one another to observe distributions
# Shortlist those features with extremely low variance because they are unable to explain the changes in sale price
f, ax = plt.subplots(cat_cols.shape[0], cat_cols.shape[1], sharey=True, figsize=(20,cat_cols.shape[0]*10))

for r, row in enumerate(cat_cols):
    for i, col in enumerate(row):
        sns.countplot(x=house[col], ax=ax[r,i])

low_var = ['street','utilities','roofmatl','heating','landcontour','landslope']
# remove these low variance columns
house.drop(labels=low_var, axis=1, inplace=True) 

house.info()

# list of fixed column labels
fixed_features = ['mszoning','lotfrontage','lotarea','lotshape','lotconfig','masvnrtype','masvnrarea','fireplaces',
                 'neighborhood','bldgtype','housestyle','foundation','totalbsmtsf','grlivarea','garagetype',
                 'garagearea', 'paveddrive','mosold','yrsold','condition','bsmtfinsf','bsmtunfsf','totdeckporchsf',
                 'highqualfinsf_perc','age','age_remodel','garageage','saleprice']
len(fixed_features)

# Get fixed features from house data with Sale price
fixed_house = house[fixed_features]

# Get X train & y train
f = 'saleprice ~ ' + ' + '.join([col for col in fixed_house.columns if col != 'saleprice'])
print(f)
y, X = patsy.dmatrices(f, data=fixed_house, return_type='dataframe')

# get train data for X
X_train = X[X['yrsold[T.2010]'] != 1]
X_test = X[X['yrsold[T.2010]'] == 1]
print X_train.shape
print X_test.shape

# get train data for y
y_train = y.loc[X_train.index].values.ravel()
y_test = y.loc[X_test.index].values.ravel()
print y_train.shape
print y_test.shape

# Deal with multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["vif_factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

indep_var_mask = vif[vif['vif_factor']<=5].features.values.ravel()

Xs_train = StandardScaler().fit_transform(X_train[indep_var_mask])
Xs_test = StandardScaler().fit_transform(X_test[indep_var_mask])

# Compute Plain Vanilla Linear regression
linreg = LinearRegression()
linreg.fit(Xs_train, y_train)
linreg_pred = linreg.predict(Xs_test)
print'Linear Reg train R2:', linreg.score(Xs_train, y_train)
print'Linear Reg R2:', metrics.r2_score(y_test, linreg_pred)
print'Linear Reg RMSE:', np.sqrt(metrics.mean_squared_error(y_test, linreg_pred))

# Ridge
ridge_alphas = np.logspace(0, 5, 500)

optimal_ridge = RidgeCV(alphas=ridge_alphas, cv=10)
optimal_ridge.fit(Xs_train, y_train)

print optimal_ridge.alpha_

ridge = Ridge(alpha=optimal_ridge.alpha_)
ridge.fit(Xs_train, y_train)
ridge_pred = ridge.predict(Xs_test)
print'Ridge train R2:', ridge.score(Xs_train, y_train)
print'Ridge R2:', metrics.r2_score(y_test, ridge_pred)
print'Ridge RMSE:', np.sqrt(metrics.mean_squared_error(y_test, ridge_pred))

# Lasso
optimal_lasso = LassoCV(n_alphas=500, cv=10)
optimal_lasso.fit(Xs_train, y_train)

print optimal_lasso.alpha_

lasso = Lasso(alpha=optimal_lasso.alpha_)
lasso.fit(Xs_train, y_train)
lasso_pred = lasso.predict(Xs_test)
print'Lasso train R2:', lasso.score(Xs_train, y_train)
print'Lasso R2:', metrics.r2_score(y_test, lasso_pred)
print'Lasso RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lasso_pred))

# Get all the Non Zero Betas
lasso_coefs = pd.DataFrame({'coef':lasso.coef_,
                            'mag':np.abs(lasso.coef_),
                            'pred':X_train[indep_var_mask].columns})

lasso_coefs.sort_values('mag', inplace=True, ascending=False)

lasso_nonzerobeta_features = lasso_coefs[lasso_coefs.mag != 0]

print 'Percent variables not zeroed out:', lasso_nonzerobeta_features.shape[0]/float(lasso_coefs.shape[0]) * 100

# Elastic Net
l1_ratios = np.linspace(0.01, 1.0, 99)
optimal_enet = ElasticNetCV(l1_ratio=l1_ratios, n_alphas=500, cv=10)
optimal_enet.fit(Xs_train, y_train)
print optimal_enet.alpha_
print optimal_enet.l1_ratio_

enet = ElasticNet(alpha=optimal_enet.alpha_, l1_ratio=optimal_enet.l1_ratio_)
enet.fit(Xs_train, y_train)
enet_pred = enet.predict(Xs_test)
print'ElasticNet train R2:', enet.score(Xs_train, y_train)
print'ElasticNet R2:', metrics.r2_score(y_test, enet_pred)
print'EasticNet RMSE:', np.sqrt(metrics.mean_squared_error(y_test, enet_pred))

# Get all the Non Zero Betas
enet_coefs = pd.DataFrame({'coef':enet.coef_,
                            'mag':np.abs(enet.coef_),
                            'pred':X_train[indep_var_mask].columns})

enet_coefs.sort_values('mag', inplace=True, ascending=False)

enet_nonzerobeta_features = enet_coefs[enet_coefs.mag != 0]

print 'Percent variables not zeroed out:', enet_nonzerobeta_features.shape[0]/float(enet_coefs.shape[0]) * 100

# Gradient Descent Optimization, Penalty use Lasso & Ridge
sgd_params = {
    'loss':['squared_loss','huber'],
    'penalty':['l1','l2'],
    'alpha':np.logspace(-10,5,500)
}

sgd_reg = SGDRegressor()
sgd_reg_gs = GridSearchCV(sgd_reg, sgd_params, cv=10, verbose=False)

sgd_reg_gs.fit(Xs_train, y_train)

# Gradient Descent best penalty is Ridge
print sgd_reg_gs.best_params_
print sgd_reg_gs.best_score_
sgd_reg = sgd_reg_gs.best_estimator_

sgd_reg.fit(Xs_train, y_train)
sgd_reg_pred = sgd_reg.predict(Xs_test)
print'SGD Reg train R2:', sgd_reg.score(Xs_train, y_train)
print'SGD Reg R2:', metrics.r2_score(y_test, sgd_reg_pred)
print'SGD Reg RMSE:', np.sqrt(metrics.mean_squared_error(y_test, sgd_reg_pred))

value_coefs = pd.DataFrame({'coef':sgd_reg.coef_,
                            'mag':np.abs(sgd_reg.coef_),
                            'pred':X_train[indep_var_mask].columns})
value_coefs.sort_values('mag', ascending=False, inplace=True)
value_coefs.head(10)

# From Lasso
lasso_nonzerobeta_features.head(10)

# From Elastic net 
enet_nonzerobeta_features.head(10)

print'Linear Reg train R2:', np.round(linreg.score(Xs_train, y_train), decimals=4)
print'Linear Reg R2:', np.round(metrics.r2_score(y_test, linreg_pred), decimals=4)
print'Linear Reg RMSE:', np.round(np.sqrt(metrics.mean_squared_error(y_test, linreg_pred)), decimals=4)
print '----------------'
print'Ridge train R2:', np.round(ridge.score(Xs_train, y_train), decimals=4)
print'Ridge R2:', np.round(metrics.r2_score(y_test, ridge_pred), decimals=4)
print'Ridge RMSE:', np.round(np.sqrt(metrics.mean_squared_error(y_test, ridge_pred)), decimals=4)
print '----------------'
print'Lasso train R2:', np.round(lasso.score(Xs_train, y_train), decimals=4)
print'Lasso R2:', np.round(metrics.r2_score(y_test, lasso_pred), decimals=4)
print'Lasso RMSE:', np.round(np.sqrt(metrics.mean_squared_error(y_test, lasso_pred)), decimals=4)
print '----------------'
print'ElasticNet train R2:', np.round(enet.score(Xs_train, y_train), decimals=4)
print'ElasticNet R2:', np.round(metrics.r2_score(y_test, enet_pred), decimals=4)
print'EasticNet RMSE:', np.round(np.sqrt(metrics.mean_squared_error(y_test, enet_pred)), decimals=4)
print '----------------'
print'SGD Reg train R2:', np.round(sgd_reg.score(Xs_train, y_train), decimals=4)
print'SGD Reg R2:', np.round(metrics.r2_score(y_test, sgd_reg_pred), decimals=4)
print'SGD Reg RMSE:', np.round(np.sqrt(metrics.mean_squared_error(y_test, sgd_reg_pred)), decimals=4)

# Lasso Model has the best test results

data = pd.DataFrame()
data['pred'] = lasso_pred
data['true'] = y_test
data['resid'] = data['true'] - data['pred']
data['abs_resid'] = np.abs(data['resid'].values.ravel())
data['stdized_resid'] = StandardScaler().fit_transform(data[['resid']])
data.head()

# Plot prediction against actual 
sns.jointplot(x='true', y='pred', data=data)

# plot residual against pred
sns.jointplot(x='pred', y='stdized_resid', data=data)

# No apparent relationship with predictions, normally distributed

# Predict for Training data to see if there is any outliers

# predict for training data using lasso
data_train = pd.DataFrame()
lasso.fit(Xs_train, y_train)
data_train['true'] = y_train
data_train['pred'] = lasso.predict(Xs_train)
data_train['resid'] = y_train - data_train.pred
data_train['stdized_resid'] = StandardScaler().fit_transform(data_train.resid.to_frame())

sns.jointplot(x='pred', y='true', data=data_train)

sns.jointplot(x='pred', y='stdized_resid', data=data_train)

# There is 1 outlier, we will find out which row is this

temp_df = pd.concat([pd.DataFrame(Xs_train, columns=X_train[indep_var_mask].columns), data_train], axis=1)
temp_df[temp_df.pred > 13.5]

# A:
# target variable now is residual
# predictors are non-fixed features
reno_features = [x for x in house.columns if x not in fixed_features]
len(reno_features)

reno_house = house[reno_features]

f = ' ~ ' + ' + '.join([col for col in reno_house.columns])
print(f)
reno_X = patsy.dmatrix(f, data=reno_house, return_type='dataframe')

vif = pd.DataFrame()
vif["vif_factor"] = [variance_inflation_factor(reno_X.values, i) for i in range(reno_X.shape[1])]
vif["features"] = reno_X.columns

indep_var_mask = vif[vif['vif_factor']<=5].features.values.ravel()
reno_X = reno_X[indep_var_mask]

# Get y train & y test
reno_y_train = data_train.resid.values.ravel()
reno_y_test = data.resid.values.ravel()
print len(reno_y_train)
print len(reno_y_test)

# Get X train & X test
reno_X_train = reno_X.loc[X_train.index]
reno_X_test = reno_X.loc[X_test.index]
print reno_X_train.shape
print reno_X_test.shape

reno_Xs_train = StandardScaler().fit_transform(reno_X_train)
reno_Xs_test = StandardScaler().fit_transform(reno_X_test)

# Lasso
optimal_lasso = LassoCV(n_alphas=500, cv=10)
optimal_lasso.fit(reno_Xs_train, reno_y_train)

print optimal_lasso.alpha_

# Fit Lasso Model
lasso = Lasso(alpha=optimal_lasso.alpha_)
lasso.fit(reno_Xs_train, reno_y_train)
reno_lasso_pred = lasso.predict(reno_Xs_test) 
print'Lasso train Reno R2:', np.round(lasso.score(reno_Xs_train, reno_y_train), decimals=4)
print'Lasso Reno R2:', np.round(metrics.r2_score(reno_y_test, reno_lasso_pred), decimals=4)
print'Lasso Reno RMSE:', np.round(np.sqrt(metrics.mean_squared_error(reno_y_test, reno_lasso_pred)), decimals=4)

# Display the top 10 coefficients
reno_coefs = pd.DataFrame({'coef':lasso.coef_,
                            'mag':np.abs(lasso.coef_),
                            'pred':reno_X_train.columns})
reno_coefs.sort_values('mag', ascending=False, inplace=True)
reno_coefs.head(10)

# Every 1 unit increase in overall quality increase the log sale price by 0.063358

# We can use the first model to buy property that are at least 0.2009 RMSE above / below our log predicted price
# Buying below our log predicted price are undervalued properties
# The second model can be used to identify renovatable parts that increase / decrease the log sale price of the house

# First model to guide purchase decision, second model to guide renovation decisions

# However result of the renovatable model is telling us that the existing renovatable features 
# is not very good at explaining the variance from fixed features
# Only about 20% of the variance can be explained by the renovatable features

reno_data = pd.DataFrame()
reno_data['true'] = reno_y_test
reno_data['pred'] = reno_lasso_pred
reno_data['resid'] = reno_y_test - reno_lasso_pred
reno_data['stdized_resid'] = StandardScaler().fit_transform(reno_data[['resid']])

# See whether prediction from model with all features different from just fixed features
# If different means we need to make some changes to our second model,
# otherwise it means that the renovatable features dont affect price much or 
# there are features that are missing from the dataset
# In the latter case, we should just use the first model to spot investment opportunities

f = 'saleprice ~ ' + ' + '.join([col for col in house.columns if col != 'saleprice'])
print(f)
all_y, all_X = patsy.dmatrices(f, data=house, return_type='dataframe')

vif = pd.DataFrame()
vif["vif_factor"] = [variance_inflation_factor(all_X.values, i) for i in range(all_X.shape[1])]
vif["features"] = all_X.columns

indep_var_mask = vif[vif['vif_factor']<=5].features.values.ravel()
all_X = all_X[indep_var_mask]

all_X_train = all_X.loc[X_train.index]
all_X_test = all_X.loc[X_test.index]
all_Xs_train = StandardScaler().fit_transform(all_X_train)
all_Xs_test = StandardScaler().fit_transform(all_X_test)

# Lasso
optimal_lasso = LassoCV(n_alphas=500, cv=10)
optimal_lasso.fit(all_Xs_train, y_train)

print optimal_lasso.alpha_

lasso = Lasso(alpha=optimal_lasso.alpha_)
lasso.fit(all_Xs_train, y_train)
all_lasso_pred = lasso.predict(all_Xs_test) 
print'ElasticNet train Reno R2:', np.round(lasso.score(all_Xs_train, y_train), decimals=4)
print'ElasticNet Reno R2:', np.round(metrics.r2_score(y_test, all_lasso_pred), decimals=4)
print'EasticNet Reno RMSE:', np.round(np.sqrt(metrics.mean_squared_error(y_test, all_lasso_pred)), decimals=4)

# The fit is better when you use all the features,
# It might be the case that renovatable features do have a slight impact on sale price
# Lets do t test to check whether both predictions are different to confirm

# See their variances
f, ax = plt.subplots(1,2,sharey=True, sharex=True, figsize=(20,5))
ax[0].set_title('All Features Prediction')
ax[1].set_title('Fixed Features Prediction')
sns.distplot(all_lasso_pred, ax=ax[0])
sns.distplot(lasso_pred, ax=ax[1])

# Check both means and medians
print 'Mean/Median for all features: ', np.mean(all_lasso_pred), np.median(all_lasso_pred)
print 'Mean/Median for fixed features: ', np.mean(lasso_pred), np.median(lasso_pred)

# Mean and Medians are very close for both distributions
# It is safe to assume that both are normally distributed
# Assumption for t test is that both variances must be the same
print 'Variance for all features:', np.std(all_lasso_pred)
print 'Variance for fixed features:', np.std(lasso_pred)

# Standard Deviation is very close, 
# Safe to do T test for related scores
ss.ttest_rel(all_lasso_pred, lasso_pred)

# T test is above 0.05, we do not reject null hypothesis, average predictions between both are the same
# We can conclude that predicting sale prices using the fixed features model is no different from all features

# We can also conclude from this that the renovatable features have not much impact on sale price, which
# explains the low score from the renovatable features model.

# Therefore we should only use the Fixed model for spotting investment opportunities

# A:
house[house.salecondition == 'Abnorml'].shape[0] / float(house.shape[0])

# Only 6% of the data are abnormal
# Get high recall on minority, high precision on majority

# Make new binary target, 1 == Abnorml & 0 == not
target = house.salecondition.map(lambda x: 1 if x == 'Abnorml' else 0)
print 'Base Accuracy', 1 - target.mean()

f = ' ~ ' + ' + '.join([col for col in house.columns if col != 'salecondition'])
print(f)
logreg_X = patsy.dmatrix(f, data=house, return_type='dataframe')

from imblearn.over_sampling import SMOTE

logreg_X_resampled, target_resampled = SMOTE(kind='regular').fit_sample(logreg_X, target)

logreg_X_resampled = pd.DataFrame(logreg_X_resampled,columns=logreg_X.columns)
target_resampled = pd.DataFrame(target_resampled)

# train / test set by year sold in 2010 or not
logreg_X_train = logreg_X_resampled[logreg_X_resampled['yrsold[T.2010]'] == 0]
logreg_X_test = logreg_X_resampled[logreg_X_resampled['yrsold[T.2010]'] == 1]
target_train = target_resampled.loc[logreg_X_train.index].values.ravel()
target_test = target_resampled.loc[logreg_X_test.index].values.ravel()
print logreg_X_train.shape
print logreg_X_test.shape
print target_train.shape
print target_test.shape

# Deal with Collinearity
vif = pd.DataFrame()
vif["vif_factor"] = [variance_inflation_factor(logreg_X.values, i) for i in range(logreg_X.shape[1])]
vif["features"] = logreg_X.columns

indep_var_mask = vif[vif['vif_factor']<=5].features.values.ravel()
logreg_X_train = logreg_X_train[indep_var_mask]
logreg_X_test = logreg_X_test[indep_var_mask]
print logreg_X_train.shape
print logreg_X_test.shape

from sklearn.metrics import classification_report

logreg_Xs_train = StandardScaler().fit_transform(logreg_X_train)
logreg_Xs_test = StandardScaler().fit_transform(logreg_X_test)

# Plain Logistic Regression
lr = LogisticRegression()
lr.fit(logreg_Xs_train, target_train)
yhat_plain = lr.predict(logreg_Xs_test)

# Ridge Logistic Regression
lr_ridge = LogisticRegressionCV(penalty='l2', Cs=200, cv=10)
lr_ridge.fit(logreg_Xs_train, target_train)

yhat_ridge = lr_ridge.predict(logreg_Xs_test)

# Lasso Logistic Regression
lr_lasso = LogisticRegressionCV(penalty='l1', solver='liblinear', Cs=100, cv=10)
lr_lasso.fit(logreg_Xs_train, target_train)

yhat_lasso = lr_lasso.predict(logreg_Xs_test)

print classification_report(target_test, yhat_plain, labels=[1,0], target_names=['abnormal','not_abnormal'])

print classification_report(target_test, yhat_ridge, labels=[1,0], target_names=['abnormal','not_abnormal'])

print classification_report(target_test, yhat_lasso, labels=[1,0], target_names=['abnormal','not_abnormal'])

# Lasso gets best precision and recall
# get best predictors of abnormal sales type

lr_lasso_coef = pd.DataFrame({'pred':logreg_X_test.columns,
                            'coef':lr_lasso.coef_.ravel(),
                            'mag':np.abs(lr_lasso.coef_.ravel()),
                             'odds':np.exp(lr_lasso.coef_.ravel())})

lr_lasso_coef.sort_values('mag', ascending=False, inplace=True)
lr_lasso_coef.head(10)



