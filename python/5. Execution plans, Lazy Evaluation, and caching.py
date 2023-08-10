import findspark
findspark.init()
from pyspark import SparkContext
sc = SparkContext(master="local[3]")  #note that we set the number of workers to 3

get_ipython().run_cell_magic('time', '', 'RDD=sc.parallelize(range(1000000))')

RDD.toDebugString()

from math import cos
def taketime(i):
    [cos(j) for j in range(10)]
    return cos(i)

get_ipython().run_cell_magic('time', '', 'taketime(5)')

get_ipython().run_cell_magic('time', '', 'Interm=RDD.map(lambda x: taketime(x))')

print Interm.toDebugString()

get_ipython().run_cell_magic('time', '', "print 'out=',Interm.reduce(lambda x,y:x+y)")

get_ipython().run_cell_magic('time', '', "print 'out=',Interm.filter(lambda x:x>0).count()")

get_ipython().run_cell_magic('time', '', 'Interm=RDD.map(lambda x: taketime(x)).cache()')

print Interm.toDebugString()

get_ipython().run_cell_magic('time', '', "print 'out=',Interm.reduce(lambda x,y:x+y)")

get_ipython().run_cell_magic('time', '', "print 'out=',Interm.filter(lambda x:x>0).count()")

A=sc.parallelize(range(1000000))
print A.getNumPartitions()

B= A.map(lambda x: (2*x,x))     .partitionBy(10)
print B.getNumPartitions()

def getPartitionInfo(G):
    d=0
    if len(G)>1: 
        for i in range(len(G)-1):
            d+=abs(G[i+1][1]-G[i][1]) # access the glomed RDD that is now a  list
        return (G[0][0],len(G),d)
    else:
        return(None)

output=B.glom().map(lambda B: getPartitionInfo(B)).collect()
print output

A=sc.parallelize(range(1000000))    .map(lambda x:(x,x)).partitionBy(10)
print A.glom().map(len).collect()

#select 10% of the entries
B=A.filter(lambda (k,v): k%10==0)
# get no. of partitions
print B.glom().map(len).collect()

C=B.map(lambda (k,x):(x/10,x)).partitionBy(10) 
print C.glom().map(len).collect()

D=B.repartition(10)
print D.glom().map(len).collect()

