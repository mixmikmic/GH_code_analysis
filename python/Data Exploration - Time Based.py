import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('../data/All_data_cbs.csv').drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)

df.columns

# Drop tinie tempah
df = df[(df['Channel Id'] != 'UCDSX4RQN7fzIlZ1nSubwCcQ')]

def top_k_percentile_views_mask(df, k):
    top_k_percent = df['ViewCount'].nlargest(int(df.shape[0] * (k / 100.))).iloc[-1]
    return df['ViewCount'] > top_k_percent

df['PublishedAt'] = pd.to_datetime(df['PublishedAt'], errors='coerce')
df['PrevPublishedAt'] = pd.to_datetime(df['PrevPublishedAt'], errors='coerce')
df.dropna()

df['TimeDiff'] = pd.to_timedelta(df['PublishedAt']) - pd.to_timedelta(df['PrevPublishedAt'])

df = df[df['PrevViewCount'] > 0]
df['Views-PercentChange'] = (df['ViewCount'] - df['PrevViewCount']) / df['PrevViewCount'].astype(np.float)

# Drop some that have negative time diff values
df = df[df['TimeDiff'] > pd.Timedelta(0)]

df['Daysdiff'] = (df['TimeDiff'] / np.timedelta64(24, 'h')).astype(np.int)

df = df[df['ViewCount'] < 10000]

daysdiff_viewcount_mean = df.groupby('Daysdiff')['ViewCount'].mean()
plt.scatter(daysdiff_viewcount_mean.index, (daysdiff_viewcount_mean))

df_t = df[df['Views-PercentChange'] < 5]
plt.scatter(df_t['Daysdiff'], df_t['Views-PercentChange'])
plt.axhline(0, color='red')

df['clickbait-difference'] = df['Title-clickbait'] - df['PrevTitle-clickbait']

sns.distplot(df['Title-clickbait'])

sns.distplot(df['PrevTitle-clickbait'])

sns.distplot(df['clickbait-difference'])

sns.jointplot(df['clickbait-difference'], df['Views-PercentChange'])

sns.distplot(df['ViewCount'])

df = df[df['Views-PercentChange'] < 10]
sns.distplot(df['Views-PercentChange'])

df = pd.read_csv('../data/All_data_cbs.csv').drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)

# Drop Tinie Tempah
df = df[(df['Channel Id'] != 'UCDSX4RQN7fzIlZ1nSubwCcQ')]

# Get the time difference
df['PublishedAt'] = pd.to_datetime(df['PublishedAt'], errors='coerce')
df['PrevPublishedAt'] = pd.to_datetime(df['PrevPublishedAt'], errors='coerce')
df['TimeDiff'] = pd.to_timedelta(df['PublishedAt']) - pd.to_timedelta(df['PrevPublishedAt'])

# Drop all samples that didn't have a previous video
df = df[df['PrevViewCount'] > 0]
df['Views-Difference'] = df['ViewCount'] - df['PrevViewCount']
df['Views-PercentChange'] = df['Views-Difference'] / df['PrevViewCount'].astype(np.float)
df.dropna()

# Drop some that have negative time diff values 
df = df[df['TimeDiff'] > pd.Timedelta(0)]

# Get the time difference in days
df['Daysdiff'] = (df['TimeDiff'] / np.timedelta64(24, 'h')).astype(np.int)

# Drop outliers
df = df[df['ViewCount'] < 1000000]
df = df[df['Views-PercentChange'] < 100]

# Get the difference in clickbait scores
df['clickbait-difference'] = df['Title-clickbait'] - df['PrevTitle-clickbait']

df.columns

from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

features = ['subscriberCount', 'channelVideoCount', 'channelViewCount',
            'PrevCommentCount', 'PrevDislikeCount', 'PrevLikeCount', 'PrevViewCount',
            'Title-clickbait', 'PrevTitle-clickbait', 'Daysdiff', 'clickbait-difference']

X = df[features]
#X = MinMaxScaler(feature_range=(0,1)).fit_transform(X)
#y = np.log1p(df['Views-PercentChange']+2)
y = df['Views-Difference']
#m = y.min()
#y = np.log1p(y - m + 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    
reg = GradientBoostingRegressor()
reg.fit(X_train, y_train)
print "R^2: {0}".format(reg.score(X_test, y_test))

sns.set_style('whitegrid')
plt.figure(figsize=(10,12))
sns.barplot(x=reg.feature_importances_, y=features)

xgb = XGBRegressor()

xgb.fit(X_train, y_train)
xgb.score(X_test, y_test)

y_pred = xgb.predict(X_test)
plot_df = pd.DataFrame(y_pred, columns=['Predictions'])
plot_df['True'] = np.array(y_test)
sns.regplot(x='Predictions', y='True', data=plot_df)

residuals = y_pred - y_test

sns.distplot(residuals)

num_bins = 1000
bins = pd.cut(df['PrevViewCount'], num_bins)
df['bin'] = bins

features = ['subscriberCount', 'channelVideoCount', 'channelViewCount',
            'PrevCommentCount', 'PrevDislikeCount', 'PrevLikeCount', #'PrevViewCount',
            'Title-clickbait', 'PrevTitle-clickbait', 'Daysdiff', 'clickbait-difference']

for bin in set(bins):
    df_t = df[df['bin'] == bin]
    if len(df_t) < 500:
        continue
    print "Processing " + bin

    X = df_t[features]
    y = df_t['Views-Difference']
    m = y.min()
    y = np.log1p(y + np.abs(m) + 1)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    xgb = XGBRegressor(n_estimators=1000)
    xgb.fit(X_train, y_train)
    
    print xgb.score(X_test, y_test)

features = ['subscriberCount', 'channelVideoCount', 'channelViewCount',
            'PrevCommentCount', 'PrevDislikeCount', 'PrevLikeCount', 'PrevViewCount',
            'Title-clickbait', 'PrevTitle-clickbait', 'Daysdiff', 'clickbait-difference']

X = df[features]
y = df[['ViewCount', 'Views-Difference']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

xgb_total = XGBRegressor()
xgb_total.fit(X_train, y_train['ViewCount'])
xgb_total.score(X_test, y_test['ViewCount'])

xgb_diff = XGBRegressor()
xgb_diff.fit(X_train, y_train['Views-Difference'])
xgb_diff.score(X_test, y_test['Views-Difference'])

y_pred = xgb_total.predict(X_test)
plt.scatter(y_test['ViewCount'], y_pred)

residuals = y_test['ViewCount'] - y_pred
sns.distplot(residuals, kde=False)

y_diff_predictions = np.array(np.array(xgb_diff.predict(X_test)) + np.array(X_test['PrevViewCount']))

plt.scatter(y_test['ViewCount'], y_diff_predictions)

residuals = y_diff_predictions - np.array(y_test['ViewCount'])
sns.distplot(residuals, kde=False)

plt.scatter(y_test['Views-Difference'], xgb_diff.predict(X_test))

from sklearn.model_selection import cross_val_score

cv1 = cross_val_score(XGBRegressor(), X, y['ViewCount'], scoring='r2')
cv2 = cross_val_score(XGBRegressor(), X, y['Views-Difference'], scoring='r2')

print "{0} +- {1}".format(cv1.mean(), cv1.var())
print "{0} +- {1}".format(cv2.mean(), cv2.var())

cv1 = cross_val_score(XGBRegressor(), X, y['ViewCount'], scoring='neg_mean_squared_error')
cv1 = np.sqrt(-cv1)
cv2 =  cross_val_score(XGBRegressor(), X, y['Views-Difference'], scoring='neg_mean_squared_error')
cv2 = np.sqrt(-cv2)
print "{0} +- {1}".format(cv1.mean(), cv1.var())
print "{0} +- {1}".format(cv2.mean(), cv2.var())



