from IPython.display import Image
from IPython.core.display import HTML

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as py
get_ipython().run_line_magic('matplotlib', 'inline')

f=open('data_1024.csv')
a=f.readlines()
f.close()
d={}
for i in a:
    c=i.split()
    d[c[0]]=[c[1],c[2]]
    #print(d[c[0]])

l1=[]
l2=[]
l1_2=[]
for i in d:
    l1.append(d[i][0])
    l2.append(d[i][1])
    l1_2.append([d[i][0],d[i][1]])
nl1=np.array(l1_2[1:])
l1=l1[1:]
l2=l2[1:]
kmeans = KMeans(n_clusters=2, random_state=0).fit(nl1)
py.figure(figsize=(15,10))
py.plot(nl1[np.where(kmeans.labels_==0)][:,0],nl1[np.where(kmeans.labels_==0)][:,1],'r4')
py.plot(nl1[np.where(kmeans.labels_==1)][:,0],nl1[np.where(kmeans.labels_==1)][:,1],'c4')





