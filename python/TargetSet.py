get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')
matplotlib.style.use('seaborn-notebook')
plt.rcParams['figure.figsize'] = (14.0, 11.0)

import dmc

dft = dmc.loading.data_train()
dfc = dmc.loading.data_class()

dft = dft.dropna(subset=['voucherID', 'rrp', 'productGroup'])
dfc = dfc.dropna(subset=['voucherID', 'rrp', 'productGroup'])

len(dft), len(dfc)

inv_features = ['articleID', 'colorCode', 'productGroup', 'voucherID', 'customerID']

def count_entries(data, features):
    profile = []
    for feat in features:
        profile.append((feat, data.count(feat)))
    return profile

def profile_dfc(column, known, unknown):
    entries = list(dfc[column])
    known_prof = count_entries(entries, known)
    unknown_prof = count_entries(entries, unknown)
    print(len(known), ' features are known and ', len(unknown), ' features are not')
    known_amt, unknown_amt = sum([e[1] for e in known_prof]), sum([e[1] for e in unknown_prof])
    print('Ratio unknown:known for rows: ', unknown_amt/known_amt)
    print(known_amt/len(entries), '% of the rows features are known')
    print(unknown_amt/len(entries), '% of the rows features are not known')
    
def diff(column, training=dft, test=dfc):
    unknown = set(test[column]) - set(training[column])
    known = set(test[column]) - unknown
    return known, unknown
    
def diff_and_profile(column):
    known, unknown = diff(column)
    profile_dfc(column, known, unknown)

for feature in inv_features:
    print(feature)
    diff_and_profile(feature)

knownGroup, uknownGroup = diff('productGroup')
knownArt, uknownArt = diff('articleID')

# all new productGroups contain new articles. New articles caused new productGroups indeed
for group in uknownGroup:
    art_set = set(dfc.articleID[dfc.productGroup == group])
    dif = art_set - knownArt
    print(len(dif), len(art_set))

known = []
for art in unknownArt:
    group = set(dfc.productGroup[dfc.articleID == art]).pop()
    if group in knownGroup:
        known.append(group)
len(known), set(known)

knownVoucher, uknownVoucher = diff('voucherID')
usedVouchers = set(dfc.voucherID[dfc.voucherID.isin(knownVoucher)])

returned_articles = dft[dft.voucherID.isin(knownVoucher)].groupby(['voucherID']).returnQuantity.sum()
bought_articles = dft[dft.voucherID.isin(knownVoucher)].groupby(['voucherID']).quantity.sum()
voucher_return_probs = returned_articles / bought_articles
axes = voucher_return_probs.sort_values().plot('bar')
axes.set_ylim([0,1])

summedVoucher = dfc[dfc.voucherID.isin(knownVoucher)].groupby(['voucherID']).agg('count')
summedVoucher.orderID

kArticle, ukArticle = diff('articleID')
kProdG, ukProdG = diff('productGroup')
kCustomer, ukCustomer = diff('customerID')
kVoucher, ukVoucher = diff('voucherID')

df = pd.DataFrame(dfc.index)

df['knownArticle'] = dfc.articleID.apply(lambda x: 0 if x in ukArticle else 1)
df['knownProdGrp'] = dfc.productGroup.apply(lambda x: 0 if x in ukProdG else 1)
df['knownCustomer'] = dfc.customerID.apply(lambda x: 0 if x in ukCustomer else 1)
df['knownVoucher'] = dfc.voucherID.apply(lambda x: 0 if x in ukVoucher else 1)
df.groupby(['knownArticle', 'knownProdGrp', 'knownCustomer', 'knownVoucher']).agg('count')

import process as p
dfm = p.processed_data()
dfm.orderDate = pd.to_datetime(dfm.orderDate)

start, end, split = pd.Timestamp('2014-1-1'), pd.Timestamp('2014-12-31'), pd.Timestamp('2014-10-1')
mask = (dfm.orderDate >= start) & (dfm.orderDate <= end)
df_full = dfm[mask]
df_train = df_full[df_full.orderDate < split]
df_class = df_full[df_full.orderDate >= split]
len(df_train), len(df_class), len(df_full), len(dfm)

kArticle, ukArticle = diff('articleID', df_train, df_class)
kProdG, ukProdG = diff('productGroup', df_train, df_class)
kCustomer, ukCustomer = diff('customerID', df_train, df_class)
kVoucher, ukVoucher = diff('voucherID', df_train, df_class)
dfv = pd.DataFrame(index=df_class.index, data=df_class.index)
dfv['knownArticle'] = df_class.articleID.apply(lambda x: 0 if x in ukArticle else 1)
dfv['knownProdGrp'] = df_class.productGroup.apply(lambda x: 0 if x in ukProdG else 1)
dfv['knownCustomer'] = df_class.customerID.apply(lambda x: 0 if x in ukCustomer else 1)
dfv['knownVoucher'] = df_class.voucherID.apply(lambda x: 0 if x in ukVoucher else 1)
dfv.groupby(['knownArticle', 'knownProdGrp', 'knownCustomer', 'knownVoucher']).agg('count')

