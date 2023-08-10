get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import requests
import numpy as np
import seaborn as sns

from lob_data_utils.roc_results import results_10000 as results
from lob_data_utils import lob

response = requests.get('http://localhost:8000/result/')
rec = []
for r in response.json():
    if r.get('data_length') == 5050:
        svm = r.get('algorithm').get('svm')
        rec.append({
            'kernel': svm.get('kernel'),
            'c': svm.get('c'),
            'gamma': svm.get('gamma'),
            'coef0': svm.get('coef0'),
            'roc_auc_score': r.get('roc_auc_score'),
            'stock': r.get('stock')
        })
df = pd.DataFrame(rec)

len(df)

log_res = []
for i, row in df.iterrows():
    log_res.append(results.get(row['stock']))
df['log_res'] = log_res
df['diff'] = df['roc_auc_score'] - log_res

print(len(df['stock'].unique()))
print(len(df[df['log_res'] < df['roc_auc_score']]['stock'].unique()))

print(df[df['log_res'] >= df['roc_auc_score']]['stock'].unique())

df[df['log_res'] <= df['roc_auc_score']]

df.groupby('stock').aggregate({'roc_auc_score': np.max}).head()

bests = []
df_best_agg = df.groupby('stock', as_index=False)['diff'].idxmax()
df_bests = df.loc[df_best_agg]
df_bests.index = df_bests['stock']

df_bests.groupby('kernel')['kernel'].count().plot(kind='bar')

df_bests['diff'].plot(kind='hist')

df_bests['diff'].plot(kind='bar')

df_bests.groupby(['kernel', 'c', 'gamma'])[['coef0']].count().plot(kind='bar')

df_bests.groupby(['kernel', 'c', 'gamma', 'coef0'])[['coef0']].count().plot(kind='bar')

# TODO: mean square error by kernel for the bests? or for all

print(df_bests[df_bests['kernel'] == 'rbf']['diff'].median())
print(df_bests[df_bests['kernel'] == 'sigmoid']['diff'].median())
print(df_bests[df_bests['kernel'] == 'linear']['diff'].median())

print(df_bests[df_bests['kernel'] == 'rbf']['diff'].std())
print(df_bests[df_bests['kernel'] == 'sigmoid']['diff'].std())
print(df_bests[df_bests['kernel'] == 'linear']['diff'].std())

print(df_bests[df_bests['kernel'] == 'rbf']['diff'].mean())
print(df_bests[df_bests['kernel'] == 'sigmoid']['diff'].mean())
print(df_bests[df_bests['kernel'] == 'linear']['diff'].mean())

print(df_bests[df_bests['kernel'] == 'rbf'][df_bests['diff'] < 0]['diff'].mean())
print(df_bests[df_bests['kernel'] == 'sigmoid'][df_bests['diff'] < 0]['diff'].mean())
print(df_bests[df_bests['kernel'] == 'linear'][df_bests['diff'] < 0]['diff'].mean())

print(df_bests[df_bests['kernel'] == 'rbf'][df_bests['diff'] < 0]['diff'].min())
print(df_bests[df_bests['kernel'] == 'sigmoid'][df_bests['diff'] < 0]['diff'].min())
print(df_bests[df_bests['kernel'] == 'linear'][df_bests['diff'] < 0]['diff'].min())


df_roc = pd.DataFrame()
df_roc['stock'] = results.keys()
df_roc['roc_area'] = results.values()

df_roc = df_roc.sort_values(by='roc_area', ascending=False)

dfs = {}
dfs_test = {}

stocks = df_roc['stock'].values

for s in stocks:
    d, d_test = lob.load_prepared_data(s, data_dir='../data/prepared/', length=5050)
    dfs[s] = d
    dfs_test[s] = d_test

df_roc = pd.DataFrame()
df_roc['stock'] = [s for s in results.keys() if s in stocks]
df_roc['roc_area'] = [results[s] for s in results.keys() if s in stocks]
df_roc = df_roc.sort_values(by='roc_area', ascending=False)
df_roc.head()

df_summary = pd.DataFrame(index=stocks)
sum_sell_ask_mean = []
sum_buy_bid_mean = []
max_trade_price = []
min_trade_price = []
bid_ask_spread = []
pearson_corrs1 = []
pearson_corrs2 = []
bid_len = []
ask_len = []

from scipy.stats import pearsonr
for s in stocks:
    sum_sell_ask_mean.append(dfs[s]['sum_sell_ask'].mean())
    sum_buy_bid_mean.append(dfs[s]['sum_buy_bid'].mean())
    max_trade_price.append(max( dfs[s]['bid_price'].max(), dfs[s]['ask_price'].max()))
    min_trade_price.append(max( dfs[s]['bid_price'].min(), dfs[s]['ask_price'].min()))
    bid_ask_spread.append((dfs[s]['ask_price'] - dfs[s]['bid_price']).mean())
    p1, p2 = pearsonr(dfs[s]['queue_imbalance'], dfs[s]['mid_price'])
    pearson_corrs1.append(p1)
    pearson_corrs2.append(p2)
    max_len_bid = 0
    max_len_ask = 0
    for i, row in dfs[s].iterrows():
        if len(row['bid']) > max_len_bid:
            max_len_bid = len(row['bid'])
        if len(row['ask']) > max_len_ask:
            max_len_ask = len(row['ask'])

    bid_len.append(max_len_bid)
    ask_len.append(max_len_ask)
df_summary['roc_area'] = df_roc['roc_area'].values
df_summary['sum_sell_ask_mean'] = sum_sell_ask_mean
df_summary['sum_buy_bid_mean'] = sum_buy_bid_mean
df_summary['diff_mean_bid_ask'] = df_summary['sum_sell_ask_mean'] - df_summary['sum_buy_bid_mean']
df_summary['max_trade_price'] = max_trade_price
df_summary['min_trade_price'] = min_trade_price
df_summary['diff_trade_price'] = df_summary['max_trade_price'] - df_summary['min_trade_price']
df_summary['bid_ask_spread'] = bid_ask_spread
df_summary['pearson_corr1'] = pearson_corrs1
df_summary['pearson_corr2'] = pearson_corrs2
df_summary['len_ask'] = ask_len
df_summary['len_bid'] = bid_len

df_summary.sort_values(by='bid_ask_spread')[df_summary['roc_area'] >= 0.58].head()

df_bests.sort_index(inplace=True)
df_summary.sort_index(inplace=True)
df_all = df_bests.join(df_summary)
diff_ind = []
for i, r in df_bests.iterrows():
    if r['diff'] <= 0:
        diff_ind.append(0)
    else:
        diff_ind.append(1)
df_all['diff_ind'] = diff_ind

df_all.head()

df_summary.head()

df_summary.info()

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(["sigmoid", "linear", "rbf"])
df_all['kernel_class'] = le.transform(df_all['kernel'])
df_all.head()

features = ['c', 'coef0', 'gamma', 'roc_auc_score', 'diff', 
            'diff_mean_bid_ask', 'diff_trade_price', 'diff_ind',
            'bid_ask_spread', 'len_ask', 'len_bid', 'kernel_class']

sns.heatmap(df_all[features].corr(), annot=True)

data_lens = []
for s in df_all.index:
    data_lens.append(len(dfs[s]))
df_all['data_len'] = data_lens
df_all.head()

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
a_features = ['len_bid', 'len_ask', 'bid_ask_spread',
            'max_trade_price', 'min_trade_price', 'sum_sell_ask_mean',
           'sum_buy_bid_mean']

features = ['len_bid', 'len_ask', 
           'sum_sell_ask_mean', 'sum_buy_bid_mean']

X = df_all[features]

kmeans = KMeans(n_clusters=2, random_state=None).fit_predict(X)
df_all['del'] = kmeans

df_all.groupby(['diff_ind', 'del'])['del'].count().plot(kind='bar')


sns.heatmap(df_all[features + ['del', 'diff_ind', 'kernel_class', 'roc_auc_score' ]].corr(), 
            annot=True)

features = ['c', 'coef0', 'gamma', 'roc_auc_score', 'diff', 
            'diff_mean_bid_ask', 'diff_trade_price', 'diff_ind',
            'bid_ask_spread', 'len_ask', 'len_bid', 'kernel_class', 'del']
sns.heatmap(df_all[features].corr(), annot=True)

from sklearn.cluster import SpectralClustering, MeanShift, AffinityPropagation 
from sklearn.decomposition import PCA
a_features = ['len_bid', 'len_ask', 'bid_ask_spread', 
            'max_trade_price', 'min_trade_price', 'sum_sell_ask_mean',
           'sum_buy_bid_mean']
features = [  'bid_ask_spread', 
            ]

X = df_all[features]

kmeans = MeanShift().fit_predict(X)
df_all['del1'] = kmeans
df_all.groupby(['diff_ind', 'del1'])['del1'].count().plot(kind='bar')





f = ['sum_sell_ask_mean', 'sum_buy_bid_mean',
       'diff_mean_bid_ask', 'max_trade_price', 'min_trade_price',
       'diff_trade_price', 'bid_ask_spread', 'pearson_corr1', 'pearson_corr2',
       'len_ask', 'len_bid', 'data_len']
print(len(df_all[f][df_all['diff'] < 0]))
df_all[f][df_all['diff'] < 0]

df_all[f].describe()



