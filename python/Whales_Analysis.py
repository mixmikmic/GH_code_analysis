sc.stop()

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from pyspark import SparkContext
sc = SparkContext(master=master_url)

from pyspark.sql import Row, SQLContext,DataFrame
from pyspark.sql.types import *

sqlContext = SQLContext(sc)

get_ipython().magic('pylab inline')

#!pip install pandas
#!pip install scipy

import pandas as pd
import datetime as dt

from scipy.io import loadmat,savemat,whosmat

from string import split
from collections import Counter
import re
import numpy as np
from numpy import shape

from glob import glob
from time import time

import sys
sys.path.append('lib')
from row_parser import *

Parse_rules,field_names,RowObject = init_parser_parameters()

from pyspark.sql import DataFrame

CVS_Data=sc.textFile("/CVS/")
row=CVS_Data.first()
print row

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

#parse(row)

RDD=CVS_Data.map(parse)
# RDD.take(3)

df=sqlContext.createDataFrame(RDD)
df.show()

t0=time()
print df.cache().count()
print time()-t0

t0=time()
print df.count()
time()-t0

from row_parser import unpackArray
import numpy
def g(row):
    #return numpy.array([1,2]) #
    return unpackArray(row.MSP,data_type=numpy.float64)
def unpackArray(x,data_type=numpy.int16):
    return numpy.frombuffer(x,dtype=data_type)
L=df.take(20)
for a in L:
    plot(g(a))

spectra=df.map(g)
type(spectra.first())

spectra.cache().count()

from time import time
from spark_PCA import *

t0=time()
COV=computeCov(spectra)
print time()-t0



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

df

get_ipython().run_cell_magic('writefile', 'lib/row_parser.py', 'import numpy\nfrom pyspark.sql import Row, SQLContext,DataFrame\nfrom pyspark.sql.types import *\nimport datetime as dt\n\ndef packArray(a):\n    if type(a)!=numpy.ndarray:\n        raise Exception("input to packArray should be numpy.ndarray. It is instead "+str(type(a)))\n    return bytearray(a.tobytes())\ndef unpackArray(x,data_type=numpy.int16):\n    return numpy.frombuffer(x,dtype=data_type)\n\ndef init_parser_parameters():\n    def parse_date(s):\n        return dt.datetime.strptime(s,\'%Y-%m-%d %H:%M:%S.%f\')\n    def parse_array(a):\n        np_array=numpy.array([numpy.float64(x) for x in a])\n        return packArray(np_array)\n    def parse_int(s):\n        return int(s)\n    def parse_float(s):\n        return float(s)\n    def parse_string(s):\n        return(s)\n\n    Fields=[(\'time\', \'datetime\'),\n        (\'species\', \'str\'),\n        (\'site\', \'str\'),\n        (\'rec_no\', \'str\'),\n        (\'bout_i\', \'int\'),\n        (\'peak2peak\', \'float\'),\n        (\'MSN\', \'array\',202),\n        (\'MSP\', \'array\',101),\n        (\'TPWS1\', \'bool\'),\n        (\'MD1\', \'bool\'),\n        (\'FD1\', \'bool\'),\n        (\'TPWS2\', \'bool\'),\n        (\'MD2\', \'bool\'),\n        (\'FD2\', \'bool\'),\n        (\'TPWS3\', \'bool\'),\n        (\'MD3\', \'bool\'),\n        (\'FD3\', \'bool\')]\n\n    global Parse_rules, RowObject\n    #prepare date structure for parsing\n    Parse_rules=[]\n    index=0\n    for field in Fields:\n        _type=field[1]\n        #print _type\n        _len=1 # default length in terms of csv fields\n        if _type ==\'array\': \n            parser=parse_array\n            _len=int(field[2])\n        elif _type==\'datetime\': \n            parser=parse_date\n        elif _type==\'int\': \n            parser=parse_int\n        elif _type==\'float\': \n            parser=parse_float\n        elif _type==\'bool\': \n            parser=parse_int\n        elif _type==\'str\': \n            parser=parse_string\n        else:\n            print \'unrecognized type\',_type\n        rule={\'name\':field[0],\n              \'start\':index,\n              \'end\':index+_len,\n              \'parser\':parser}\n        print field,rule\n        Parse_rules.append(rule)\n        index+=_len\n\n    field_names=[a[\'name\'] for a in Parse_rules]\n    RowObject= Row(*field_names)\n    return Parse_rules,field_names,RowObject\n\ndef parse(row):\n    global Parse_rules, RowObject\n    items=row.split(\',\')\n    D=[]\n    for pr in Parse_rules:\n        start=pr[\'start\']\n        end=pr[\'end\']\n        parser=pr[\'parser\']\n        if end-start==1:\n            D.append(parser(items[start]))\n        else:\n            D.append(parser(items[start:end]))\n    return RowObject(*D)')

get_ipython().system('')

