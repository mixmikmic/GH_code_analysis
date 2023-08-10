import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

df=pd.read_csv('College_Data',index_col=0)

df.head()

df.info()

df.describe()

sns.lmplot(x='Room.Board', y='Grad.Rate', data=df, hue='Private',fit_reg=False)

sns.lmplot(x='Outstate', y='F.Undergrad', data=df, hue='Private',fit_reg=False)

g=sns.FacetGrid(df, hue='Private',size=6,aspect=2)
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)

g=sns.FacetGrid(df, hue='Private',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)

df[df['Grad.Rate'] > 100]

df['Grad.Rate']['Cazenovia College'] = 100

g = sns.FacetGrid(df,hue="Private",size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)

kmeans.fit(df.drop('Private',axis=1))

kmeans.cluster_centers_

def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0

df['Cluster']=df['Private'].apply(converter)

df.head()

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))

