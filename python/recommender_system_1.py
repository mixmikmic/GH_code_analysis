from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import binarycf as bcf ## you can find this in this repository!

get_ipython().magic('matplotlib inline')
from pylab import rcParams
rcParams['figure.figsize'] = 10, 5 ## width, height (inches)

df_customers = pd.read_csv('purchase-data/customers.txt',sep='\t')[['customerid','householdid']]
df_orderlines = pd.read_csv('purchase-data/orderlines.txt',sep='\t')[['orderid','productid']]
df_orders = pd.read_csv('purchase-data/orders.csv',sep='\t')[['customerid','orderid']]
df_products = pd.read_csv('purchase-data/products.txt',sep='\t')[['PRODUCTID','PRODUCTGROUPNAME']]

#dg = pd.read_csv('purchase-data/orders.csv',sep='\t')
#dg[dg.customerid==0].groupby('customerid').size()

df = pd.merge(df_orderlines,df_orders,on='orderid') ## add customerid
df = pd.merge(df,df_customers,on='customerid')      ## add householdid
df = pd.merge(df,df_products,left_on='productid',right_on='PRODUCTID') ## add PRODUCTGROUPNAME
df = df[['householdid','PRODUCTGROUPNAME','productid']].copy()
df.rename(columns={'householdid':'customer','PRODUCTGROUPNAME':'category','productid':'item'},
          inplace=True)
df.category = df.category.fillna('OTHER') ## fill NA
df.to_csv('df_long.csv',index=False)
df.head(5)

## How many items are in each category?
df.groupby(['category','item']).size().reset_index().groupby('category').size()

## number of customers in each category
df.groupby(['category','customer']).size().reset_index().groupby('category').size()

## In how many categories a customer buy products?
df.groupby(['customer','category']).size().reset_index().groupby('customer').size().value_counts()

## the wide table of customer x category
df_cats = df[['customer','category']].copy()
df_cats['val'] = 1
df_cats = df_cats.pivot_table(index='customer', columns='category', values='val',
                              aggfunc=lambda x: 1, fill_value=0)
df_cats.iloc[5:10,:]

df_catxcat = pd.concat([df_cats[df_cats[cat]==1].apply(np.sum) for cat in df_cats.columns],axis=1)
df_catxcat.columns = df_cats.columns
df_catxcat

df_common = pd.merge(bcf.customer_in_cat(df,'APPAREL'), bcf.customer_in_cat(df,'ARTWORK'),
                     on='customer')
df_common.head()

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df_common, df_common.ARTWORK,
                                                 test_size=0.2, random_state=4)
df_train = X_train.copy() # data frame containing customer IDs
df_test = X_test.copy() 
X_train = df_train.APPAREL # series of sets of products in APPAREL
X_test = df_test.APPAREL

popular_items = bcf.get_popular_items(y_train)
popular_items ## top 5 items in ARTWORK

df_compare = pd.DataFrame(y_train)
df_compare['success'] = df_compare.ARTWORK.apply(lambda r: len(r.intersection(popular_items))>0)
df_compare.tail()

## using popular items 
y_test.apply(lambda x: len(x.intersection(popular_items))>0).mean()

df_tmp = defaultdict(list)
df_head = pd.DataFrame(X_train[:10]).reset_index()
for i in range(df_head.shape[0]):
    row = df_head.iloc[i]
    for x in row[1]:
        df_tmp['customer'].append(row[0])
        df_tmp['item'].append(x)

df_tmp = pd.DataFrame(df_tmp,columns=['customer','item'])
df_tmp['val'] = '1'
df_tmp.pivot(index='customer',columns='item',values='val').fillna('0')

yhat = X_test.apply(lambda x: bcf.ubcf(x,X_train,y_train,k=30)) ## make a recommendation
df_compare = pd.concat([yhat,y_test],axis=1).rename(
                columns={'APPAREL':'Recommendation','ARTWORK':'Purchased_item'})
df_compare['success'] = df_compare.apply(lambda r:len(set.intersection(r[0],r[1]))>0,axis=1)
df_compare.tail()

## User-based collaborative filtering
bcf.success_rate(yhat,y_test)

df_tmp.pivot(index='customer',columns='item',values='val').fillna('0') ## the same as before!

similar_ibcfs = bcf.compute_similar_items(df_train,k=10) ## step 1

yhat = X_test.apply(lambda x: bcf.ibcf(x,similar_ibcfs)) ## make a recommendation
bcf.success_rate(yhat,y_test)

df_result_cf = pd.read_csv('recom_result.csv')

top_cf = df_result_cf.sort_values(by='cv',ascending=False).groupby(['cat1','cat2']).first()
top_cf

top_cf.method.value_counts()

df_result_cf[df_result_cf.method!='popular'].sort_values(by='cv',ascending=False).groupby(['cat1','cat2','method']).first().reset_index().groupby('method')['metric'].value_counts()

ctab = df_result_cf[df_result_cf.method=='popular'].pivot(index='cat1',columns='cat2',values='cv').fillna(0)
sns.heatmap(ctab,annot=True);

ctab = df_result_cf[(df_result_cf.method=='UBCF') & (df_result_cf.k==50) & (df_result_cf.metric=='cosine')].pivot(index='cat1',columns='cat2',values='cv').fillna(0)
sns.heatmap(ctab,annot=True);

ctab = df_result_cf[(df_result_cf.method=='IBCF') & (df_result_cf.k==50) & (df_result_cf.metric=='common')].pivot(index='cat1',columns='cat2',values='cv').fillna(0)
sns.heatmap(ctab,annot=True);

