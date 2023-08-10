import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, RobustScaler

## load data
products = pd.read_csv('cproducts.csv')
tender = pd.read_csv('ctender.csv')

## check shape of files
print('product file has {} rows and {} columns'.format(products.shape[0], products.shape[1]))
print('tender file has {} rows and {} columns'.format(tender.shape[0], tender.shape[1]))

# this data file contains product level information of transactions made by customers
products.head()

# this file contains payment mode information used by customers in their transactions
tender.head()

## fill missing values

products['promotion_description'].fillna('no_promo', inplace=True)
products['Gender'].fillna('no_gender', inplace=True)
products['State'].fillna('no_state', inplace=True)
products['PinCode'].fillna(-1, inplace=True)
products['DOB'].fillna("1", inplace=True)

## convert data into numeric / float

for c in products.columns:
    lbl = LabelEncoder()
    if products[c].dtype == 'object' and c not in ['store_description','customerID','transactionDate']:
        products[c] = lbl.fit_transform(products[c])

## scaling, creating matrix and running k-means

stores = list(set(products['store_code']))

cluster_labels = []
cluster_store = []
cluster_data = []
cluster_customers = []
cluster_score = []

for x in stores:
    cld = products[products['store_code'] == x]
    cluster_customers.append(cld['customerID'])
    cld.drop(['store_code','customerID','transactionDate','store_description'], axis=1, inplace=True)
    
    rbs = RobustScaler()
    cld2 = rbs.fit_transform(cld)
    
    km1 = KMeans(n_clusters=3)
    km2 = km1.fit(cld2)
    label = km2.predict(cld2)
    
    s_score = silhouette_score(cld2, label)
    cluster_score.append(s_score)
    
    cluster_labels.append(label)
    cluster_store.append(np.repeat(x, cld.shape[0]))
    cluster_data.append(cld2)

# check mean score per store
np.mean(cluster_score)

## merge list into ndarray
cluster_data = np.concatenate(cluster_data)

## check if the array has same rows as products file - Yes!
cluster_data.shape

## convert nested lists as 1d array
cluster_customers = np.concatenate(cluster_customers)
cluster_store = np.concatenate(cluster_store)
cluster_labels = np.concatenate(cluster_labels)

## create submission files
sub1 = pd.DataFrame({'customerID':cluster_customers, 'store_code':cluster_store, 'cluster':cluster_labels})

np.savetxt('../subOne_18.txt', cluster_data)
sub1.to_csv('../subtwo_18.csv', index=False)

