import findspark
findspark.init()
from pyspark import SparkContext

#sc.stop()
sc = SparkContext(master="local[6]",pyFiles=['lib/spark_PCA.py'])

from pyspark.sql import *
sqlContext = SQLContext(sc)

get_ipython().magic('pylab inline')
import sys
sys.path.append('./lib')

import numpy as np
from spark_PCA import computeCov

# read the file in a dataframe.
df=sqlContext.read.csv('SP500.csv',header='true',inferSchema='true')
df.count()

columns=df.columns
col=[c for c in columns if '_D' in c]


# Add code to extract tickers here
tickers=[c[:-2] for c in col]

tickers[:10],len(tickers)

def make_array(row):
    # Complete function as described above
    arr = np.zeros(len(tickers))
    trans = row.asDict()
    idx = 0
    
    for ticker in tickers:
        val = trans[ticker+'_D']
        if val and not np.isnan(val):
            arr[idx] = val
        idx+=1            
    return arr
    
###-----FILL-IN using make_array-----    
Rows= df.rdd.map(lambda x:make_array(x))
Rows.first()[:20]

# We are now ready to run ComputeCov to create the covariance matrix.
OUT=computeCov(Rows)
OUT.keys()

from numpy import linalg as LA
eigval,eigvec=LA.eigh(OUT['Cov'])
eigval=eigval[-1::-1] # reverse order
eigvec=eigvec[:,-1::-1]

# Add code to plot here
def plot_var_explained():
    k=50
    plot((list(cumsum(eigval[:k])))/sum(eigval))
    title('% of Variance Explained')
    ylabel('% of Variance')
    xlabel('Num. of Eigenvector')
    grid()
plot_var_explained()

plot(eigvec[:,0])

import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))
plt.scatter(eigvec.T[1], eigvec.T[2])
plt.title('Scatter plot projecting 2 largest variance eigen-vectors')
plt.show()

import pickle
D=pickle.load(open('Tickers.pkl','rb'))
TickerInfo=D['TickerInfo']
tickers=D['Tickers']
TickerInfo.head()

# list all companies in the Energy sector
TickerInfo[TickerInfo['SECTOR_ID']=='EN']

#Exclude the following stocks from the dataset
exclude = ['GOOG', 'STT', 'SEE', 'CI', 'EFX']
tickers = [x for x in tickers if x not in exclude]

def calc_sectors(tickers):
    # Your code here
    sectors = []
    
    for i in range(len(tickers)):
        tab = TickerInfo[TickerInfo['Ticker']==tickers[i]]
        if len(tab)==1:
            sectors.append(tab['SECTOR_ID'].values[0])
        else:
            sectors.append(tickers[i])
    
    return sectors
sectors = calc_sectors(tickers)

# Create a dictionary that maps each ticker to the corresponding basis vector
Tick_rep={}
for i in range(len(tickers)):
    Tick_rep[tickers[i]]=eigvec[i,:]

len(sectors)

from sklearn.neighbors import KNeighborsClassifier

d=20
k=5
T='HAL'
def find_closest(T,d=10,k=10):
    nbclf = KNeighborsClassifier(k, metric='euclidean') 
    nbclf.fit([eigs[0:d] for eigs in Tick_rep.values()], Tick_rep.keys())
    dist, ind = nbclf.kneighbors(Tick_rep[T][:d], k)

    gg = 0
    for i in ind[0]: 
        stockID = Tick_rep.keys()[i]
        idx = tickers.index(stockID)
        print(tickers[idx], sectors[idx], (dist[0][gg])**2) #squared Euclidean distance
        gg += 1
        
find_closest('BAC')



import Tester
Tester.test0(tickers)

import Tester
Tester.test1(eigval,eigvec)

import Tester
Tester.test2(eigval,eigvec)

import Tester
Tester.test3(eigval,eigvec)

import Tester
Tester.test4(eigval,eigvec)

import Tester
Tester.test5(find_closest)

import Tester
Tester.test6(find_closest)

import Tester
Tester.test7(find_closest)

import Tester
Tester.test8(find_closest)

import Tester
Tester.test9(find_closest)

import Tester
Tester.test10(find_closest)

import Tester
Tester.test11(find_closest)

import Tester
Tester.test12(find_closest)

import Tester
Tester.test13(find_closest)

import Tester
Tester.test14(find_closest)



