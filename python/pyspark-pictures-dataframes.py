# versions
import IPython
print("pyspark version:" + str(sc.version))
print("Ipython version:" + str(IPython.__version__))

# agg
x = sqlContext.createDataFrame([("Alice","Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.agg({"amt":"avg"})
x.show()
y.show()

# alias
from pyspark.sql.functions import col
x = sqlContext.createDataFrame([("Alice","Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.alias('transactions')
x.show()
y.show()
y.select(col("transactions.to")).show()

# cache
x = sqlContext.createDataFrame([("Alice","Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
x.cache()
print(x.count()) # first action materializes x in memory
print(x.count()) # later actions avoid IO overhead

# coalesce
x_rdd = sc.parallelize([("Alice","Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)],2)
x = sqlContext.createDataFrame(x_rdd, ['from','to','amt'])
y = x.coalesce(numPartitions=1)
print(x.rdd.getNumPartitions())
print(y.rdd.getNumPartitions())

# collect
x = sqlContext.createDataFrame([("Alice","Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.collect() # creates list of rows on driver
x.show()
print(y)

# columns
x = sqlContext.createDataFrame([("Alice","Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.columns #creates list of column names on driver
x.show()
print(y)

# corr
x = sqlContext.createDataFrame([("Alice","Bob",0.1,0.001),("Bob","Carol",0.2,0.02),("Carol","Dave",0.3,0.02)], ['from','to','amt','fee'])
y = x.corr(col1="amt",col2="fee")
x.show()
print(y)

# count
x = sqlContext.createDataFrame([("Alice","Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
x.show()
print(x.count())

# cov
x = sqlContext.createDataFrame([("Alice","Bob",0.1,0.001),("Bob","Carol",0.2,0.02),("Carol","Dave",0.3,0.02)], ['from','to','amt','fee'])
y = x.cov(col1="amt",col2="fee")
x.show()
print(y)

# crosstab
x = sqlContext.createDataFrame([("Alice","Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.crosstab(col1='from',col2='to')
x.show()
y.show()

# cube
x = sqlContext.createDataFrame([("Alice","Bob",0.1),("Alice","Carol",0.2)], ['from','to','amt'])
y = x.cube('from','to')
x.show()
print(y) # y is a grouped data object, aggregations will be applied to all numerical columns
y.sum().show() 
y.max().show()

# describe
x = sqlContext.createDataFrame([("Alice","Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
x.show()
x.describe().show()

# distinct
x = sqlContext.createDataFrame([("Alice","Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3),("Bob","Carol",0.2)], ['from','to','amt'])
y = x.distinct()
x.show()
y.show()

# drop
x = sqlContext.createDataFrame([("Alice","Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.drop('amt')
x.show()
y.show()

# dropDuplicates / drop_duplicates
x = sqlContext.createDataFrame([("Alice","Bob",0.1),("Bob","Carol",0.2),("Bob","Carol",0.3),("Bob","Carol",0.2)], ['from','to','amt'])
y = x.dropDuplicates(subset=['from','to'])
x.show()
y.show()

# dropna
x = sqlContext.createDataFrame([(None,"Bob",0.1),("Bob","Carol",None),("Carol",None,0.3),("Bob","Carol",0.2)], ['from','to','amt'])
y = x.dropna(how='any',subset=['from','to'])
x.show()
y.show()

# dtypes
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.dtypes
x.show()
print(y)

# explain
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
x.show()
x.agg({"amt":"avg"}).explain(extended = True)

# fillna
x = sqlContext.createDataFrame([(None,"Bob",0.1),("Bob","Carol",None),("Carol",None,0.3)], ['from','to','amt'])
y = x.fillna(value='unknown',subset=['from','to'])
x.show()
y.show()

# filter
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.filter("amt > 0.1")
x.show()
y.show()

# first
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.first()
x.show()
print(y)

# flatMap
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.flatMap(lambda x: (x[0],x[2])) 
print(y) # implicit coversion to RDD
y.collect()

# foreach
from __future__ import print_function

# setup
fn = './foreachExampleDataFrames.txt' 
open(fn, 'w').close()  # clear the file
def fappend(el,f):
    '''appends el to file f'''
    print(el,file=open(f, 'a+') )

# example
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.foreach(lambda x: fappend(x,fn)) # writes into foreachExampleDataFrames.txt
x.show() # original dataframe
print(y) # foreach returns 'None'
# print the contents of the file
with open(fn, "r") as foreachExample:
    print (foreachExample.read())

# foreachPartition
from __future__ import print_function

#setup
fn = './foreachPartitionExampleDataFrames.txt'
open(fn, 'w').close()  # clear the file
def fappend(partition,f):
    '''append all elements in partition to file f'''
    print([el for el in partition],file=open(f, 'a+'))

x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
x = x.repartition(2) # force 2 partitions
y = x.foreachPartition(lambda x: fappend(x,fn)) # writes into foreachPartitionExampleDataFrames.txt

x.show() # original dataframe
print(y) # foreach returns 'None'
# print the contents of the file
with open(fn, "r") as foreachExample:
    print (foreachExample.read())

# freqItems
x = sqlContext.createDataFrame([("Bob","Carol",0.1),                                 ("Alice","Dave",0.1),                                 ("Alice","Bob",0.1),                                 ("Alice","Bob",0.5),                                 ("Carol","Bob",0.1)],                                ['from','to','amt'])
y = x.freqItems(cols=['from','amt'],support=0.8)
x.show()
y.show()

# groupBy
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Alice","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.groupBy('from')
x.show()
print(y)

# groupBy(col1).avg(col2)
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Alice","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.groupBy('from').avg('amt')
x.show()
y.show()

# head
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.head(2)
x.show()
print(y)

# intersect
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Alice",0.2),("Carol","Dave",0.1)], ['from','to','amt'])
z = x.intersect(y)
x.show()
y.show()
z.show()

# isLocal
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.isLocal()
x.show()
print(y)

# join
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = sqlContext.createDataFrame([('Alice',20),("Bob",40),("Dave",80)], ['name','age'])
z = x.join(y,x.to == y.name,'inner').select('from','to','amt','age')
x.show()
y.show()
z.show()

# limit
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.limit(2)
x.show()
y.show()

# map
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.map(lambda x: x.amt+1)
x.show()
print(y.collect())  # output is RDD

# mapPartitions
def amt_sum(partition):
    '''sum the value in field amt'''
    yield sum([el.amt for el in partition])
    
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
x = x.repartition(2) # force 2 partitions
y = x.mapPartitions(lambda p: amt_sum(p))
x.show()
print(x.rdd.glom().collect()) # flatten elements on the same partition
print(y.collect())
print(y.glom().collect())

# na
x = sqlContext.createDataFrame([(None,"Bob",0.1),("Bob","Carol",None),("Carol",None,0.3),("Bob","Carol",0.2)], ['from','to','amt'])
y = x.na  # returns an object for handling missing values, supports drop, fill, and replace methods
x.show()
print(y)
y.drop().show()
y.fill({'from':'unknown','to':'unknown','amt':0}).show()
y.fill(0).show()

# orderBy
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.orderBy(['from'],ascending=[False])
x.show()
y.show()

# persist
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
x.persist(storageLevel=StorageLevel(True,True,False,True,1)) # StorageLevel(useDisk,useMemory,useOffHeap,deserialized,replication=1)
x.show()
x.is_cached

# printSchema
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
x.show()
x.printSchema()

# randomSplit
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.randomSplit([0.5,0.5])
x.show()
y[0].show()
y[1].show()

# rdd
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.rdd
x.show()
print(y.collect())

# registerTempTable
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
x.registerTempTable(name="TRANSACTIONS")
y = sqlContext.sql('SELECT * FROM TRANSACTIONS WHERE amt > 0.1')
x.show()
y.show()

# repartition
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.repartition(3)
print(x.rdd.getNumPartitions())
print(y.rdd.getNumPartitions())

# replace
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.replace('Dave','David',['from','to'])
x.show()
y.show()

# rollup
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.rollup(['from','to'])
x.show()
print(y) # y is a grouped data object, aggregations will be applied to all numerical columns
y.sum().show()
y.max().show()

# sample
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.sample(False,0.5)
x.show()
y.show()

# sampleBy
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Alice","Carol",0.2),("Alice","Alice",0.3),                                ('Alice',"Dave",0.4),("Bob","Bob",0.5),("Bob","Carol",0.6)],                                 ['from','to','amt'])
y = x.sampleBy(col='from',fractions={'Alice':0.1,'Bob':0.9})
x.show()
y.show()

# schema
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.schema
x.show()
print(y)

# select
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.select(['from','amt'])
x.show()
y.show()

# selectExpr
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.selectExpr(['substr(from,1,1)','amt+10'])
x.show()
y.show()

# show
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
x.show()

# sort
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Alice",0.3)], ['from','to','amt'])
y = x.sort(['to'])
x.show()
y.show()

# sortWithinPartitions
x = sqlContext.createDataFrame([('Alice',"Bob",0.1,1),("Bob","Carol",0.2,2),("Carol","Alice",0.3,2)],                                ['from','to','amt','p_id']).repartition(2,'p_id')
y = x.sortWithinPartitions(['to'])
x.show()
y.show()
print(x.rdd.glom().collect()) # glom() flattens elements on the same partition
print(y.rdd.glom().collect())

# stat
x = sqlContext.createDataFrame([("Alice","Bob",0.1,0.001),("Bob","Carol",0.2,0.02),("Carol","Dave",0.3,0.02)], ['from','to','amt','fee'])
y = x.stat
x.show()
print(y)
print(y.corr(col1="amt",col2="fee"))

# subtract
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.1)], ['from','to','amt'])
z = x.subtract(y)
x.show()
y.show()
z.show()

# take
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.take(num=2)
x.show()
print(y)

# toDF
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.toDF("seller","buyer","amt")
x.show()
y.show()

# toJSON
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Alice",0.3)], ['from','to','amt'])
y = x.toJSON()
x.show()
print(y.collect())

# toPandas
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.toPandas()
x.show()
print(type(y))
y

# unionAll
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2)], ['from','to','amt'])
y = sqlContext.createDataFrame([("Bob","Carol",0.2),("Carol","Dave",0.1)], ['from','to','amt'])
z = x.unionAll(y)
x.show()
y.show()
z.show()

# unpersist
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
x.cache()
x.count()
x.show()
print(x.is_cached)
x.unpersist()
print(x.is_cached)

# where (filter)
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.where("amt > 0.1")
x.show()
y.show()

# withColumn
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",None),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.withColumn('conf',x.amt.isNotNull())
x.show()
y.show()

# withColumnRenamed
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.withColumnRenamed('amt','amount')
x.show()
y.show()

# write
import json
x = sqlContext.createDataFrame([('Alice',"Bob",0.1),("Bob","Carol",0.2),("Carol","Dave",0.3)], ['from','to','amt'])
y = x.write.mode('overwrite').json('./dataframeWriteExample.json')
x.show()
# read the dataframe back in from file
sqlContext.read.json('./dataframeWriteExample.json').show()

