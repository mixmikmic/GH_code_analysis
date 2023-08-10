import pandas as pd
import numpy as np
import sklearn as sk
import urllib
import math
get_ipython().magic('pylab inline')

import findspark
findspark.init()

from pyspark import SparkContext
#sc.stop()
sc = SparkContext(master="local[3]",pyFiles=['lib/numpy_pack.py','lib/spark_PCA.py','lib/computeStats.py'])

from pyspark import SparkContext
from pyspark.sql import *
sqlContext = SQLContext(sc)

import sys
sys.path.append('./lib')

import numpy as np
from numpy_pack import packArray,unpackArray
from spark_PCA import computeCov
from computeStats import computeOverAllDist, STAT_Descriptions

### Read the data frame from pickle file

data_dir='../../Data/Weather'
file_index= 'BSBSSSBB'

from pickle import load

#read statistics
filename=data_dir+'/STAT_%s.pickle'%file_index
STAT,STAT_Descriptions = load(open(filename,'rb'))
print 'keys from STAT=',STAT.keys()

#read data
filename=data_dir+'/US_Weather_%s.parquet'%file_index

df=sqlContext.read.parquet(filename)
print df.count()
df.show(10)

### Function to extract the values 
measures = ['TOBS', 'TMIN', 'TMAX', 'PRCP', 'SNOW', 'SNWD']
data = []
for i in measures:
    sqlContext.registerDataFrameAsTable(df,'weather')
    Query="SELECT station, vector FROM weather\n\tWHERE measurement='%s'"%(i)
    df1 = sqlContext.sql(Query)
    rows=df1.rdd.map(lambda row:[unpackArray(row['vector'],np.float16), row['station']]).collect()
    measure = pd.DataFrame(rows, columns=('vector', 'station'))
    data.append(measure)

for i in range(0,4):
    data[i]['Jan_%s' %measures[i]] = data[i]['vector'].apply(lambda x: np.nanmean(x[:31])/10)
    data[i]['Feb_%s' %measures[i]] = data[i]['vector'].apply(lambda x: np.nanmean(x[31:59])/10)
    data[i]['Mar_%s' %measures[i]] = data[i]['vector'].apply(lambda x: np.nanmean(x[59:90])/10)
    data[i]['Apr_%s' %measures[i]] = data[i]['vector'].apply(lambda x: np.nanmean(x[90:120])/10)
    data[i]['May_%s' %measures[i]] = data[i]['vector'].apply(lambda x: np.nanmean(x[120:151])/10)
    data[i]['Jun_%s' %measures[i]] = data[i]['vector'].apply(lambda x: np.nanmean(x[151:181])/10)
    data[i]['Jul_%s' %measures[i]] = data[i]['vector'].apply(lambda x: np.nanmean(x[181:212])/10)
    data[i]['Aug_%s' %measures[i]] = data[i]['vector'].apply(lambda x: np.nanmean(x[212:243])/10)
    data[i]['Sep_%s' %measures[i]] = data[i]['vector'].apply(lambda x: np.nanmean(x[243:273])/10)
    data[i]['Oct_%s' %measures[i]] = data[i]['vector'].apply(lambda x: np.nanmean(x[273:304])/10)
    data[i]['Nov_%s' %measures[i]] = data[i]['vector'].apply(lambda x: np.nanmean(x[304:334])/10)
    data[i]['Dec_%s' %measures[i]] = data[i]['vector'].apply(lambda x: np.nanmean(x[334:])/10)
    data[i] = data[i].groupby('station').mean()
for i in range(4,6):
    data[i]['Jan_%s' %measures[i]] = data[i]['vector'].apply(lambda x: np.nanmean(x[:31]))
    data[i]['Feb_%s' %measures[i]] = data[i]['vector'].apply(lambda x: np.nanmean(x[31:59]))
    data[i]['Mar_%s' %measures[i]] = data[i]['vector'].apply(lambda x: np.nanmean(x[59:90]))
    data[i]['Apr_%s' %measures[i]] = data[i]['vector'].apply(lambda x: np.nanmean(x[90:120]))
    data[i]['May_%s' %measures[i]] = data[i]['vector'].apply(lambda x: np.nanmean(x[120:151]))
    data[i]['Jun_%s' %measures[i]] = data[i]['vector'].apply(lambda x: np.nanmean(x[151:181]))
    data[i]['Jul_%s' %measures[i]] = data[i]['vector'].apply(lambda x: np.nanmean(x[181:212]))
    data[i]['Aug_%s' %measures[i]] = data[i]['vector'].apply(lambda x: np.nanmean(x[212:243]))
    data[i]['Sep_%s' %measures[i]] = data[i]['vector'].apply(lambda x: np.nanmean(x[243:273]))
    data[i]['Oct_%s' %measures[i]] = data[i]['vector'].apply(lambda x: np.nanmean(x[273:304]))
    data[i]['Nov_%s' %measures[i]] = data[i]['vector'].apply(lambda x: np.nanmean(x[304:334]))
    data[i]['Dec_%s' %measures[i]] = data[i]['vector'].apply(lambda x: np.nanmean(x[334:]))
    data[i] = data[i].groupby('station').mean()

print data[0].shape, data[1].shape, data[2].shape, data[3].shape, data[4].shape, data[5].shape

data = pd.concat([data[0], data[1], data[2], data[3], data[4], data[5]], axis=1, join='inner')
print data.shape
data.head()

import seaborn as sns

num_plots = 12
fig, axes = plt.subplots(4,3,figsize=(18,24))

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
for i in range(len(months)):
    a = data.filter(regex=months[i])
    mat = a.cov() # to get a heatmap of the covariance matrix
#     mat = a.corr() # to get a heatmap of the correlation matrix
    row = i // 3
    col = i % 3
    ax_curr = axes[row, col]
    sns.heatmap(mat, vmax=1, square = True, ax=ax_curr)
    ax_curr.set_title('%s Comparisons' %months[i])



