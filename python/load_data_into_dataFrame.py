sc.stop()

from pyspark import SparkContext

sc = SparkContext(master=master_url)

from pyspark.sql import Row, SQLContext,DataFrame
from pyspark.sql.types import *

sqlContext = SQLContext(sc)

get_ipython().magic('pylab inline')

import pandas as pd
import datetime as dt

from scipy.io import loadmat,savemat,whosmat

from string import split
from collections import Counter
import re
import numpy as np
from numpy import shape

from glob import glob


Fields_string="""(time       , datetime),
(species	   , str),
(site	   , str),
(rec_no	   , str),
(bout_i	   , int),
(peak2peak  , float),
(MSN	   , array,202),
(MSP	   , array,101  ),
(TPWS1	   , bool),
(MD1	   , bool),
(FD1	   , bool),
(TPWS2	   , bool),
(MD2	   , bool),
(FD2	   , bool),
(TPWS3	   , bool),
(MD3	   , bool),
(FD3	   , bool)"""
import re
pattern=re.compile(r'\(([\.\w]*)\s*,\s*([\,\.\w]*)\s*\)')

for line in Fields_string.split('\n'):
    #print '\n',line
    match=pattern.search(line)
    if match:
        print "('%s', '%s'),"%(match.group(1),match.group(2))
    else:
        print 'no match'

Fields=[('time', 'datetime'),
('species', 'str'),
('site', 'str'),
('rec_no', 'str'),
('bout_i', 'int'),
('peak2peak', 'float'),
('MSN', 'array',202),
('MSP', 'array',101),
('TPWS1', 'bool'),
('MD1', 'bool'),
('FD1', 'bool'),
('TPWS2', 'bool'),
('MD2', 'bool'),
('FD2', 'bool'),
('TPWS3', 'bool'),
('MD3', 'bool'),
('FD3', 'bool')]

Fields

get_ipython().magic('cd /root/ipython/BeakedWhaleClassification/')
get_ipython().magic('run Credentials.ipynb')

s3helper.set_credential(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)

s3helper.open_bucket('while-classification')
s3helper.ls_s3()

dirs=s3helper.ls_s3('CVS')
dirs[:10]

from time import time
time()

t1=time()
s3helper.s3_to_hdfs('CVS', 'CVS')
time()-t1

#!/root/ephemeral-hdfs/bin/hdfs dfs -ls /CVS/                                                             

CVS_Data=sc.textFile("/CVS/")

t0=time()
print Data.count()
print time()-t0

date_format='%Y-%m-%d %H:%M:%S.%f'
def parse_date(s):
    #print 'date string="%s"'%s
    return dt.datetime.strptime(s,date_format)
def parse_array(a):
    np_array=np.array([np.float64(x) for x in a])
    return packArray(np_array)
def parse_int(s):
    return int(s)
def parse_float(s):
    return float(s)
def parse_string(s):
    return(s)

def packArray(a):
    if type(a)!=np.ndarray:
        raise Exception("input to packArray should be numpy.ndarray. It is instead "+str(type(a)))
    return bytearray(a.tobytes())
def unpackArray(x,data_type=np.int16):
    return np.frombuffer(x,dtype=data_type)

date_format='%Y-%m-%d %H:%M:%S.%f'
#prepare date structure for parsing
Parse_rules=[]
index=0
for field in Fields:
    _type=field[1]
    #print _type
    _len=1 # default length in terms of csv fields
    if _type =='array': 
        parser=parse_array
        _len=int(field[2])
    elif _type=='datetime': 
        parser=parse_date
    elif _type=='int': 
        parser=parse_int
    elif _type=='float': 
        parser=parse_float
    elif _type=='bool': 
        parser=parse_int
    elif _type=='str': 
        parser=parse_string
    else:
        print 'unrecognized type',_type
    rule={'name':field[0],
          'start':index,
          'end':index+_len,
          'parser':parser}
    print field,rule
    Parse_rules.append(rule)
    index+=_len

field_names=[a['name'] for a in Parse_rules]
print field_names
RowObject= Row(*field_names)
RowObject

def parse(row):
    items=row.split(',')
    D=[]
    for pr in Parse_rules:
        start=pr['start']
        end=pr['end']
        parser=pr['parser']
        if end-start==1:
            D.append(parser(items[start]))
        else:
            D.append(parser(items[start:end]))
    return RowObject(*D)



from pyspark.sql import DataFrame
RDD=sc.parallelize([R])
df=sqlContext.createDataFrame(RDD)

df.printSchema()

CVS_Data=sc.textFile("/CVS/")
RDD=CVS_Data.map(parse)

df=sqlContext.createDataFrame(RDD)

t0=time()
df.cache().count()
time()-t0

t0=time()
print df.count()
time()-t0

import sys
sys.path.append('lib')

import spark_PCA

# %load lib/spark_PCA.py
import numpy as np
from numpy import linalg as LA

def outerProduct(X):
    """Computer outer product and indicate which locations in matrix are undefined"""
    O=np.outer(X,X)
    N=1-np.isnan(O)
    return (O,N)

def sumWithNan(M1,M2):
    """Add two pairs of (matrix,count)"""
    (X1,N1)=M1
    (X2,N2)=M2
    N=N1+N2
    X=np.nansum(np.dstack((X1,X2)),axis=2)
    return (X,N)

def computeCov(RDDin):
    """computeCov recieves as input an RDD of np arrays, all of the same length, 
    and computes the covariance matrix for that set of vectors"""
    RDD=RDDin.map(lambda v:np.insert(v,0,1)) # insert a 1 at the beginning of each vector so that the same 
                                           #calculation also yields the mean vector
    OuterRDD=RDD.map(outerProduct)   # separating the map and the reduce does not matter because of Spark uses lazy execution.
    (S,N)=OuterRDD.reduce(sumWithNan)
    # Unpack result and compute the covariance matrix
    # print 'RDD=',RDD.collect()
    # print 'shape of S=',S.shape,'shape of N=',N.shape
    # print 'S=',S
    # print 'N=',N
    E=S[0,1:]
    NE=np.float64(N[0,1:])
    print 'shape of E=',E.shape,'shape of NE=',NE.shape
    Mean=E/NE
    O=S[1:,1:]
    NO=np.float64(N[1:,1:])
    Cov=O/NO - np.outer(Mean,Mean)
    # Output also the diagnal which is the variance for each day
    Var=np.array([Cov[i,i] for i in range(Cov.shape[0])])
    return {'E':E,'NE':NE,'O':O,'NO':NO,'Cov':Cov,'Mean':Mean,'Var':Var}

if __name__=="__main__":
    # create synthetic data matrix with 10 rows and rank 2
    
    V=2*(np.random.random([2,10])-0.5)
    data_list=[]
    for i in range(1000):
        f=2*(np.random.random(2)-0.5)
        data_list.append(np.dot(f,V))
    # compute covariance matrix
    RDD=sc.parallelize(data_list)
    OUT=computeCov(RDD)

    #find PCA decomposition
    eigval,eigvec=LA.eig(OUT['Cov'])
    #print 'eigval=',eigval
    #print 'eigvec=',eigvec

spectra=df.map(lambda row:unpackArray(row.MSP,data_type=np.float64))

t0=time()
COV=computeCov(spectra)
print time()-t0

#COV

M=COV['Mean']
S=np.sqrt(COV['Var'])
plot(M-S)
plot(M)
plot(M+S)

plot(np.sqrt(COV['Var']))

eigval,eigvec=LA.eig(COV['Cov'])

eigval=eigval/sum(eigval)
sum(eigval)

plot(cumsum(eigval[:10]))

shape(eigvec)

figure(figsize=[10,7])
for i in range(4):
    plot(eigvec[:,i],label='ev'+str(i))
legend()
grid()

sum(eigvec[:,1]**2)

#Cuviers=df.filter(df.species==u'Cuviers' & df.TPWS2==1)
Cuviers=df.filter(df.TPWS2==1).filter(df.species==u'Cuviers')

Gervais=df.filter(df.TPWS2==1).filter(df.species==u'Gervais')

V=eigvec[:,1:3] #vectors on which to project
def project(row):
    X=unpackArray(row.MSP,data_type=np.float64)
    return np.dot(X,V)

Cuvier_projections=np.array(Cuviers.sample(False,0.001).map(project).take(10000))
Gervais_projections=np.array(Gervais.sample(False,0.001).map(project).take(10000))

figure(figsize=[13,10])
scatter(Cuvier_projections[:,0],Cuvier_projections[:,1],c='r')
scatter(Gervais_projections[:,0],Gervais_projections[:,1],c='b')
xlim([-200,-50])
ylim([-50,50])

shape(Cuvier_projections)

df



