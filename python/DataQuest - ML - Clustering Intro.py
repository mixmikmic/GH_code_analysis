import pandas as pd
votes = pd.read_csv('114_congress.csv')
print(votes.head())
votes['party'].value_counts()
votes.mean()

from sklearn.metrics.pairwise import euclidean_distances
#use the euclidean_distances funciton. 
distance = euclidean_distances(votes.iloc[0,3:].values.reshape(1, -1),votes.iloc[2,3:].values.reshape(1,-1))
print(distance)

from sklearn.cluster import KMeans
mikes_kmean_model = KMeans(n_clusters=2,random_state=1)

senator_distances = mikes_kmean_model.fit_transform(votes.iloc[:,3:])
senator_distances[0:5]

labels = mikes_kmean_model.labels_
print(pd.crosstab(labels,votes['party']))

D_outliers = votes[(labels == 1) & (votes['party']=='D')]
D_outliers

import matplotlib.pyplot as plt
plt.scatter(x=senator_distances[:,0],y=senator_distances[:,1],c=labels)
plt.show()

extremism = (senator_distances ** 3).sum(axis=1)
votes['extremism'] = extremism
votes=votes.sort_values('extremism',ascending = False)
votes[0:10]

