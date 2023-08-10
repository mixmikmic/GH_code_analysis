import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import missingno as msno
import statsmodels.api as sm
import statsmodels.formula.api as smf

get_ipython().magic('matplotlib inline')

colors = sns.crayon_palette(['Tickle Me Pink', 'Atomic Tangerine', 'Fuzzy Wuzzy'])

# load dataset
train_df = pd.read_csv("./Sberbank/train.csv", parse_dates=['timestamp'], index_col=False)
test_df = pd.read_csv("./Sberbank/test.csv", parse_dates=['timestamp'], index_col=False)
macro_df = pd.read_csv("./Sberbank/macro.csv", parse_dates=['timestamp'], index_col=False)
train_df = pd.merge(train_df, macro_df, how='left', on='timestamp')
test_df = pd.merge(test_df, macro_df, how='left', on='timestamp')

train_df['month'] = train_df['timestamp'].dt.month
train_df['day'] = train_df['timestamp'].dt.day
train_df['year'] = train_df['timestamp'].dt.year


# Outliers/Transformations
y_train = pd.DataFrame(train_df['price_doc'])
X_train = pd.DataFrame(train_df.loc[:, train_df.columns != 'price_doc'])
# X_test = test.values

test_df['month'] = test_df['timestamp'].dt.month
test_df['day'] = test_df['timestamp'].dt.day
test_df['year'] = test_df['timestamp'].dt.year

frames = [train_df, test_df]

df = pd.concat(frames)

from sklearn import model_selection

X_train_all, X_test_all, y_train_all, y_test_all = model_selection.train_test_split(
                                                                df.loc[:, df.columns != 'price_doc'], 
                                                                df['price_doc'], test_size=1.0/5, random_state=0)

# rescale data
def rescale(feature):
    return feature.values.reshape(-1,1)

# standardize data
from sklearn.preprocessing import StandardScaler

def standardize(feature):
    scaler = StandardScaler().fit(feature)
    return scaler.transform(feature)

# normalize data
from sklearn.preprocessing import Normalizer

def normalize(feature):
    scaler = Normalizer().fit(feature)
    return scaler.transform(feature)

def reshape_feature(feature):
    preprocess_df = rescale(feature)
    preprocess_df = standardize(feature)
    preprocess_df = normalize(feature)
    return preprocess_df[0]

date_range = [train_df['timestamp'].min(),train_df['timestamp'].max()]
print date_range


macro_df['timestamp'].max()
macro_df['timestamp'].min()

date_range = [test_df['timestamp'].min(),test_df['timestamp'].max()]
date_range

time_price = train_df.loc[:, ['timestamp','price_doc']]
time_price = time_price.set_index('timestamp')

price_per_day = train_df.loc[:, ['timestamp','price_doc']].groupby('timestamp').mean()
price_per_day = (price_per_day - price_per_day.mean()) / (price_per_day.max() - price_per_day.min())
gdp_day = train_df.loc[:, ['timestamp','gdp_quart']].groupby('timestamp').mean()
gdp_day = (gdp_day - gdp_day.mean()) / (gdp_day.max() - gdp_day.min())
cpi = train_df.loc[:, ['timestamp','cpi']].groupby('timestamp').mean()
cpi = (cpi - cpi.mean()) / (cpi.max() - cpi.min())




plt.figure(figsize=(25, 15))
plt.plot(price_per_day.resample('M').sum())
plt.plot(gdp_day)
plt.plot(np.log1p(cpi))
plt.show()

by_month = train_df.loc[:,['month','year','price_doc']]
by_month = by_month.reset_index()

price_per_month = train_df.groupby(['month','year']).mean()['price_doc']

filtered_data = msno.nullity_filter(train_df, filter='bottom', n=50, p=0.999) # or filter='top'
msno.matrix(filtered_data)

important_macro_features = ['gdp_quart', 'cpi', 'ppi', 'usdrub', 'eurrub', 
                            'gdp_annual', 'rts', 'micex', 'micex_cbi_tr', 'deposits_rate', 
                            'mortgage_rate', 'income_per_cap', 'salary', 'labor_force', 
                            'unemployment', 'employment']

macro_feature_df = train_df.loc[:, important_macro_features]

df_important = df.loc[:, ['timestamp', 'full_sq',
                          'life_sq', 'floor', 'max_floor', 'material',
                          'build_year', 'num_room',
                          'kitch_sq', 'state',
                          'product_type', 'sub_area',
                          'indust_part', 'school_education_centers_raion',
                          'sport_objects_raion', 'culture_objects_top_25_raion',
                          'oil_chemistry_raion', 'metro_min_avto',
                          'green_zone_km', 'industrial_km',
                          'kremlin_km', 'radiation_km',
                          'ts_km', 'fitness_km',
                          'stadium_km', 'additional_education_km',
                          'cafe_count_1500_price_500', 'cafe_count_1500_price_high',
                          'cafe_count_2000_price_2500', 'trc_sqm_5000',
                          'cafe_count_5000', 'cafe_count_5000_price_high',
                          'gdp_quart', 'cpi',
                          'ppi', 'usdrub',
                          'eurrub', 'gdp_annual',
                          'rts', 'micex',
                          'micex_cbi_tr', 'deposits_rate',
                          'mortgage_rate', 'income_per_cap',
                          'salary', 'labor_force',
                          'unemployment', 'employment', 'price_doc']]


y_train = pd.DataFrame(df_important['price_doc']).values
X_train = pd.DataFrame(df_important.loc[:, df_important.columns != 'price_doc']).values

missing_impt_features = ['full_sq', 'life_sq', 'product_type', 'floor', 'max_floor', 
'material', 'build_year','num_room', 'kitch_sq', 'state', 'price_doc']

missing_df = df.loc[:, missing_impt_features]

filtered_data = msno.nullity_filter(missing_df, filter='bottom', n=12, p=0.5) # or filter='top'
msno.matrix(missing_df)

df[df['product_type'].isnull()]

(df.loc[df['product_type'].notnull() & df['price_doc'].notnull()].corr()).to_csv('./correlation_table.csv')


lifesq_floor = df.loc[df['life_sq'].notnull() & df['price_doc'].notnull(), ['life_sq','price_doc']]

lifesq_floor[['life_sq','price_doc']].corr()

fullsq_floor = df.loc[df['full_sq'].notnull() & df['price_doc'].notnull(), ['full_sq','price_doc']]

fullsq_floor[['full_sq','price_doc']].corr()

np.sum(df.loc[df['full_sq'].isnull(), 'life_sq'].isnull())

sns.pointplot(train_df['gdp_quart'],train_df['price_doc'])

msno.matrix(macro_feature_df)

corr = macro_feature_df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(25, 15))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

corr2 = train_df[['gdp_quart','gdp_annual']].corr()

corr2

corr3 = train_df[['gdp_quart','gdp_annual','price_doc']].corr()

corr3

def trans_plot(df, x, y, fns):    
#     ulimit = np.percentile(train_df[y].values, 98.5)
#     llimit = np.percentile(train_df[y].values, 1.5)
#     train_df[y].loc[train_df[y]>ulimit] = ulimit
#     train_df[y].loc[train_df[y]<llimit] = llimit
    
#     ulimit = np.percentile(train_df[x].values, 98.5)
#     llimit = np.percentile(train_df[x].values, 1.5)
#     train_df[x].loc[train_df[x]>ulimit] = ulimit
#     train_df[x].loc[train_df[x]<llimit] = llimit
    
    f, axes = plt.subplots(2, 3, figsize=(12, 12), sharex=True, sharey=True)
    for fn in fns:
        sns.jointplot(fn(reshape_feature(x)).values, y.values, kind="reg", dropna=True)
        plt.ylabel('{0}'.format(y), fontsize=10)
        plt.xlabel('{1}'.format(y), fontsize=10)              
    return f.tight_layout()

df.loc[df['build_year'].isnull()]

fns = [lambda x: x, np.square, np.sqrt,
       np.log, np.log1p, np.log10]

# x_='gdp_quart'
# y_='price_doc'

# trans_plot(train_df, 'gdp_quart', 'price_doc', fns)
# print [fn(reshape_feature(train_df['gdp_quart'])) for fn in fns]

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn import model_selection

# This dataset is way too high-dimensional. Better do PCA:
pca = PCA(n_components=1)

# Maybe some original features where good, too?
selection = SelectKBest(k=3)

# Build estimator from PCA and Univariate selection:

combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

# Use combined features to transform dataset:
X_features = combined_features.fit(X, y).transform(X)

svm = SVC(kernel="linear")

# Do grid search over k, n_components and C:

pipeline = Pipeline([("features", combined_features), ("svm", svm)])
pipeline.fit(X, y)
print(pipeline)

np.sum(macro_df['employment'].isnull())

log_price = np.log1p(df.loc[df['price_doc'].notnull(), 'price_doc'])
price = df.loc[df['price_doc'].notnull(), 'price_doc']

g = sns.JointGrid(data=price, size=15)
g = g.plot(sns.regplot, sns.distplot)



df.groupby('sub_area').mean()

train_df[['full_sq', 'price_doc']].corr()

X_train_all['full_sq']

df[['kremlin_km', 'sadovoe_km', 'price_doc']].sample(20)





