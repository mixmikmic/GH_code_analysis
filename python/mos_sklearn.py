get_ipython().magic('matplotlib inline')
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns; sns.set(color_codes=True); sns.set(); sns.set_context(rc={'lines.markeredgewidth': 1})

# reading data from file
nwp = pd.DataFrame.from_csv('./merged_gfs.csv', index_col='time')
obs = pd.DataFrame.from_csv('./seatac_weather.csv')

# converting timestring into a python datetime object
obs['time'] = obs.apply(lambda x: x['Date'] + ' ' + x['Time'], axis=1)
obs['time'] = obs['time'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M'))

# dropping unecessary observation columns columns
obs.index = obs.time
obs.drop(['Date', 'time', 'Time'], inplace=True, axis=1)

obs.head(5)

# resample to alight to end of hour
obs = obs.resample(dt.timedelta(hours=1), how='mean', label='right')
obs.head(5)

nwp.head(3)

# join nwp and observation on datetime
df = nwp.join(obs)

# specifying the target variable
target = 'Temperature'
nwp_feature = 'temp_height:100_lat:47.5_lon:237.5'

# plot a timeseries of temperature from model and obs 
df[[nwp_feature, target]][700:900].plot(figsize=(20,5))

# create a month variable to facet by month and dropping NA's
df['month'] = df.index.month
df = df.dropna()

# plotting the distribution of 2 variables using seaborn
sns.distplot(df[nwp_feature])
sns.distplot(df[target])

# plotting for a specific month
sns.distplot(df[nwp_feature][df['month']==2])
sns.distplot(df[target][df['month']==2])

# computing errors and faceting errors by month
errs = pd.DataFrame()
errs['hourly_errors'] = df[nwp_feature] - df[target]
errs['month'] = df['month']

g = sns.FacetGrid(errs, col="month", col_wrap=3, size=4, xlim=(-15,10))
g.map(sns.distplot, "hourly_errors");

from sklearn.linear_model import LinearRegression

# a method to split data into testing and training set every k-weeks
def split_dataframe(df, target, k=4):
    df['weekOfYear'] = df.index.weekofyear
    df_trn = df[df['weekOfYear']%k!=0]
    df_tst = df[df['weekOfYear']%4==0]

    drop_vars = ['Temperature', 'Dewpoint', 'Relhum', 'Speed', 'month', 'weekOfYear']
    if target not in drop_vars:
        drop_vars.append(target)
        
    xtrn, ytrn = df_trn.drop(drop_vars, axis=1), df_trn[target]
    xtst, ytst = df_tst.drop(drop_vars, axis=1), df_tst[target]
    
    return xtrn, ytrn, xtst, ytst

xtrn, ytrn, xtst, ytst = split_dataframe(df, target, k=4)
# set normalize to True since linear models are sensitive to scale
linear_model_mos = LinearRegression(normalize=True)
# fit data on training set
linear_model_mos.fit(xtrn, ytrn)

# look at model coefficients 
linear_model_mos.coef_

# predict on testing set
ymos = linear_model_mos.predict(xtst)

sns.kdeplot(ymos)
sns.kdeplot(ytst)

error_mos = ymos-ytst
error_raw = xtst[nwp_feature]-ytst

print "model output bias is %.2f while mos bias is %.2f" % (np.mean(error_raw), np.mean(error_mos))
print "model output rmse is %.2f while mos rmse is %.2f" % (np.mean(error_raw**2)**0.5, np.mean(error_mos**2)**0.5)

df_verif = pd.DataFrame()
df_verif['obs'] = ytst
df_verif['ymos'] = ymos
df_verif['yraw'] = xtst[nwp_feature]

idx = df_verif.index <= dt.datetime(2015,2,23)
plt.figure(figsize=(15,5))
plt.plot(df_verif.index[idx], df_verif['obs'][idx], color='k', marker='+')
plt.plot(df_verif.index[idx], df_verif['ymos'][idx], color='b', marker='+')
plt.plot(df_verif.index[idx], df_verif['yraw'][idx], color='g', marker='+')
plt.legend(['obs', 'ymos', 'yraw'])

from sklearn.ensemble import GradientBoostingRegressor

nonlinear_model_mos = GradientBoostingRegressor()
nonlinear_model_mos.fit(xtrn, ytrn)

nonlinear_model_mos

print nonlinear_model_mos.estimators_
tree_id1 = nonlinear_model_mos.estimators_[34][0]

from sklearn.tree import export_graphviz
from IPython.display import Image

with open('tree_1.dot', 'w') as dotfile:
     export_graphviz(tree_id1,
                     dotfile,
                     feature_names=xtrn.columns)

get_ipython().system('dot -Tpng tree_1.dot -o tree.png')
Image(filename='tree.png') 

idx_feature_importance = np.argsort(-nonlinear_model_mos.feature_importances_)
print xtrn.columns[idx_feature_importance][:10]

ymos_nonlinear = nonlinear_model_mos.predict(xtst)

sns.kdeplot(ymos_nonlinear)
sns.kdeplot(ytst)

error_mos_nonlinear = ymos_nonlinear-ytst

print "nonlinear mos bias is %.2f while mos bias is %.2f" % (np.mean(error_mos_nonlinear), np.mean(error_mos))
print "nonlinear mos rmse is %.2f while mos rmse is %.2f" % (np.mean(error_mos_nonlinear**2)**0.5, np.mean(error_mos**2)**0.5)

df_verif['ymos2'] = ymos_nonlinear

plt.figure(figsize=(15,5))
plt.plot(df_verif.index[idx], df_verif['obs'][idx], color='k', marker='+')
plt.plot(df_verif.index[idx], df_verif['ymos'][idx], color='b', marker='+')
plt.plot(df_verif.index[idx], df_verif['ymos2'][idx], color='g', marker='+')
plt.legend(['obs', 'ymos', 'ymos2'])

from sklearn.neighbors import KNeighborsRegressor

knn_model_mos = KNeighborsRegressor(n_neighbors=5)
knn_model_mos.fit(xtrn, ytrn)

ymos_knn = knn_model_mos.predict(xtst)

sns.kdeplot(ymos_knn)
sns.kdeplot(ytst)

df_verif['ymos3'] = ymos_knn

plt.figure(figsize=(15,5))
plt.plot(df_verif.index[idx], df_verif['obs'][idx], color='k', marker='+')
plt.plot(df_verif.index[idx], df_verif['ymos'][idx], color='b', marker='+')
plt.plot(df_verif.index[idx], df_verif['ymos3'][idx], color='g', marker='+')
plt.legend(['obs', 'ymos', 'ymos3'])

df['freeze'] = 0
df['freeze'][df['Temperature'] < 40] = 1

df['freeze'].describe()

from sklearn.ensemble import RandomForestClassifier
target = 'freeze'

xtrn, ytrn, xtst, ytst = split_dataframe(df, target, k=4)
model_icing = RandomForestClassifier(class_weight='auto')
model_icing.fit(xtrn,ytrn)

is_icing = model_icing.predict(xtst)

df_confusion = pd.crosstab(ytst, is_icing, rownames=['Actual'], colnames=['Predicted'], margins=True)
print df_confusion

df_verif['is_icing_cont'] = 0
df_verif['is_icing_cont'][df_verif['yraw'] < 40] = 1

df_confusion = pd.crosstab(ytst, df_verif['is_icing_cont'], rownames=['Actual'], colnames=['Predicted'], margins=True)
print df_confusion



