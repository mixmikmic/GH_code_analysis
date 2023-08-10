import pandas as pd
import numpy as np

# not skipping empty values
df = pd.read_csv('census2000.csv', na_filter=False)

df.info()

df['RegDens'].describe()

df['RegDens'] = df['RegDens'].replace('', np.nan).astype(float)

import seaborn as sns
import matplotlib.pyplot as plt

# Distribution of RegDens
regdens_dist = sns.distplot(df['RegDens'].dropna())
plt.show()

# Distribution of MedHHInc
medhhinc_dist = sns.distplot(df['MedHHInc'].dropna())
plt.show()

# Distribution of MeanHHSz
meanhhsz_dist = sns.distplot(df['MeanHHSz'].dropna())

plt.show()

# Distribution of MeanHHSz, with increased number of bins. More bins = more specific distplot.
meanhhsz_dist = sns.distplot(df['MeanHHSz'].dropna(), bins=100)
plt.show()

df['HasError_MeanHHSz'] = df['MeanHHSz'] < 1

g = sns.FacetGrid(df, col='HasError_MeanHHSz')
g = g.map(plt.hist, 'MedHHInc', bins=100)

plt.show()

# before
print("Before dropping errorneous rows", len(df))

# a very easy way to drop rows with MeanHHSz values below 1
df = df[df['MeanHHSz'] >= 1]

# after
print("After dropping errorneous rows", len(df))

# Distribution of RegDens
regdens_dist = sns.distplot(df['RegDens'].dropna())
plt.show()

# Distribution of MedHHInc
medhhinc_dist = sns.distplot(df['MedHHInc'].dropna())
plt.show()

# Distribution of MeanHHSz
meanhhsz_dist = sns.distplot(df['MeanHHSz'].dropna())

plt.show()

from sklearn.preprocessing import StandardScaler
df2 = df[['MedHHInc', 'MeanHHSz', 'RegDens']]
scaler = StandardScaler()

X = df2.as_matrix()
X = scaler.fit_transform(X)

from sklearn.cluster import KMeans

# set the random state. different random state seeds might result in different centroids locations
model = KMeans(n_clusters=3, random_state=42)
model.fit(X)

# sum of intra-cluster distances
print("Sum of intra-cluster distance:", model.inertia_)

print("Centroid locations:")
for centroid in model.cluster_centers_:
    print(centroid)

# set a different n_clusters
model = KMeans(n_clusters=8, random_state=42)
model.fit(X)

# sum of intra-cluster distances
print("Sum of intra-cluster distance:", model.inertia_)

print("Centroid locations:")
for centroid in model.cluster_centers_:
    print(centroid)

model = KMeans(n_clusters=3, random_state=42).fit(X)

# assign cluster ID to each record in X
# Ignore the warning, does not apply to our case here
y = model.predict(X)
df2['Cluster_ID'] = y

# how many records are in each cluster
print("Cluster membership")
print(df2['Cluster_ID'].value_counts())

# pairplot the cluster distribution.
cluster_g = sns.pairplot(df2, hue='Cluster_ID')
plt.show()

# set a different n_clusters
model = KMeans(n_clusters=8, random_state=42)
model.fit(X)

# sum of intra-cluster distances
print("Sum of intra-cluster distance:", model.inertia_)

print("Centroid locations:")
for centroid in model.cluster_centers_:
    print(centroid)

# again, ignore the warning
y = model.predict(X)
df2['Cluster_ID'] = y

# how many in each
print("Cluster membership")
print(df2['Cluster_ID'].value_counts())

# pairplot
cluster_g = sns.pairplot(df2, hue='Cluster_ID')
plt.show()

# prepare the column and bin size. Increase bin size to be more specific, but 20 is more than enough
cols = ['MedHHInc', 'MeanHHSz', 'RegDens']
n_bins = 20

# inspecting cluster 0
print("Distribution for cluster 0")
cluster_to_inspect = 0

# create subplots
fig, ax = plt.subplots(nrows=3)
ax[0].set_title("Cluster {}".format(cluster_to_inspect))

for j, col in enumerate(cols):
    # create the bins
    bins = np.linspace(min(df2[col]), max(df2[col]), 20)
    # plot distribution of the cluster using histogram
    sns.distplot(df2[df2['Cluster_ID'] == cluster_to_inspect][col], bins=bins, ax=ax[j], norm_hist=True)
    # plot the normal distribution with a black line
    sns.distplot(df2[col], bins=bins, ax=ax[j], hist=False, color="k")

plt.tight_layout()
plt.show()

# inspecting cluster 1
print("Distribution for cluster 1")
cluster_to_inspect = 1

# again, subplots
fig, ax = plt.subplots(nrows=3)
ax[0].set_title("Cluster {}".format(cluster_to_inspect))

for j, col in enumerate(cols):
    # create the bins
    bins = np.linspace(min(df2[col]), max(df2[col]), 20)
    # plot distribution of the cluster using histogram
    sns.distplot(df2[df2['Cluster_ID'] == cluster_to_inspect][col], bins=bins, ax=ax[j], norm_hist=True)
    # plot the normal distribution with a black line
    sns.distplot(df2[col], bins=bins, ax=ax[j], hist=False, color="k")
    
plt.tight_layout()
plt.show()

# list to save the clusters and cost
clusters = []
inertia_vals = []

# this whole process should take a while
for k in range(2, 15, 2):
    # train clustering with the specified K
    model = KMeans(n_clusters=k, random_state=42, n_jobs=10)
    model.fit(X)
    
    # append model to cluster list
    clusters.append(model)
    inertia_vals.append(model.inertia_)

# plot the inertia
plt.plot(range(2,15,2), inertia_vals, marker='*')
plt.show()

from sklearn.metrics import silhouette_score

print(clusters[1])
print("Silhouette score for k=4", silhouette_score(X, clusters[1].predict(X)))

print(clusters[2])
print("Silhouette score for k=6", silhouette_score(X, clusters[2].predict(X)))

import pandas as pd

# load the bank transaction dataset
df = pd.read_csv('bank.csv')

# info and the first 10 transactions
print(df.info())
print(df.head(10))

# group by account, then list all services
transactions = df.groupby(['ACCOUNT'])['SERVICE'].apply(list)

print(transactions.head(5))

from apyori import apriori

# type cast the transactions from pandas into normal list format and run apriori
transaction_list = list(transactions)
results = list(apriori(transaction_list, min_support=0.05))

# print first 5 rules
print(results[:5])

def convert_apriori_results_to_pandas_df(results):
    rules = []
    
    for rule_set in results:
        for rule in rule_set.ordered_statistics:
            rules.append([','.join(rule.items_base), ','.join(rule.items_add), # items_base = left side of rules, items_add = right side
                         rule_set.support, rule.confidence, rule.lift]) # support, confidence and lift for respective rules
    
    return pd.DataFrame(rules, columns=['Left_side', 'Right_side', 'Support', 'Confidence', 'Lift']) # typecast it to pandas df

result_df = convert_apriori_results_to_pandas_df(results)

print(result_df.head(20))

result_df = result_df.sort_values(by='Lift', ascending=False)
print(result_df.head(10))

