import os
import sys

sys.path.append(os.environ["SPARK_HOME"] + "/python/lib/py4j-0.9-src.zip")
sys.path.append(os.environ["SPARK_HOME"] + "/python/lib/pyspark.zip")

from pyspark import SparkConf, SparkContext
from pyspark import SparkFiles
from pyspark import StorageLevel
from pyspark import AccumulatorParam

sconf = SparkConf()
sconf.setAppName("PySpark Tutorial")
sconf.setMaster("spark://snehasish-barmans-macbook.local:7077")

sc = SparkContext.getOrCreate(conf = sconf)

print sc
print sc.version

sc.parallelize([1,2, "abc", (1,2), {4,5,6}]).collect()

rdd = sc.parallelize([1,2,3,4,5,6,7,8,9,10], 3)
rdd2 = sc.parallelize(xrange(10, 20), 3)

print rdd.glom().collect() # shows data grouped by partitions
print rdd2.glom().collect()

rdd.map(lambda x: x**2).collect() # map 1-to-1 transformation, operates on every element of rdd

# functions must be self-contained, no states or access global variables
def trans1(x):
    return x**2

rdd.map(trans1).collect()

rdd.filter(lambda x: x > 5).collect() # filter

datasets = "../../../Machine_Learning/WPI_ML/datasets"
print os.path.exists(datasets)
#print os.path.realpath(datasets)

# default is hdfs filesystem, to access local files, use namespace -> file:///
textrdd = sc.textFile("file:///" + os.path.realpath(datasets) + "/Audio_Standardization_ Sentences.txt", 
                      use_unicode = False, minPartitions = 3)

textrdd.glom().collect()

textrdd.flatMap(lambda x: x.split(" ")).take(11) # 1-to-many transformation, puts in a global list

def countWordsInPartition(iterator): 
    """
    @params:
        iterator: a partition of the rdd
    """
    count = 0
    for x in iterator:
        count += len(x.split(" "))
    yield count

textrdd.mapPartitions(countWordsInPartition).collect() # same as map but operates on each chunk/partition of the rdd

rdd.sortBy(keyfunc = lambda x: x, ascending = False, numPartitions = 1).collect() # sorting
# numPartitions controls the level of parallelism

rdd.sample(withReplacement = False, fraction = 0.5, seed = 13).collect() # sampling

rdd.coalesce(1).glom().collect() # reduce no. of partitions by combining partitions from each worker, thereby minimizing network traffic

rdd.repartition(2).glom().collect() 
# increases or decreases the no. of partitions, but at the cost of more network traffic, 
# because Spark has to shuffle the data across the workers.
# Use coalesce when intent to decrease the partitions.

rdd.repartition(5).glom().collect()

rdd.union(rdd2).collect() # combines two rdds -> A u B

rdd.intersection(rdd2).collect() # intersection -> A n B

rdd.subtract(rdd2).collect() # subtract -> A - B, removes all the common elements between A and B from A and returns the rest

rdd.union(rdd2).distinct().sortBy(ascending = True, keyfunc = lambda x:x).collect() # distinct

rdd.cartesian(rdd2).take(5) # all pair combinations; creates key-value RDD

rdd.zip(rdd2).collect() # zip (same as zip() in python); creates key-value RDD

rdd.keyBy(lambda x: x % 3).collect() # keyBy (converts a normal RDD into key-value RDD based on a criteria)
# result of the criteria becomes the 'key' and the element itself becomes the 'value'.

print rdd.groupBy(lambda x: x % 3).collect() # groupBy - same as 'keyBy' but all the values of a key are grouped into an iterable
print list(rdd.groupBy(lambda x: x % 3).collect()[0][1])

file_name = "square_nums.py"
sc.addFile("./" + file_name) # All workers will download this file to their node

rdd.pipe("cat").collect() # pipe
# Use an external program for custom transformations.
# Reads data as string per partition from standard input and writes as string to standard output.

rdd.pipe(SparkFiles.get(file_name)).glom().collect() # pipe

rdd.reduce(lambda acc, x: acc+x) # reduce; operation must satisfy associative and communtative property

rdd.count() # count

rdd.take(4) # take (returns as a list; selects data from one partition, then moves to another partition as required to satisfy the limit)

rdd.takeSample(False, 5, seed = 13) # takeSample

rdd.takeOrdered(4, key = lambda x:x) # takeOrdered

rdd.collect()

rdd.first() # first

rdd.top(4, key = int) # top ; returns top n items in descending order

rdd.countApprox(1000, 0.5) # countApprox

rdd.countApproxDistinct(0.7) # number of distinct elements

def showValues(x):
    print "hello: " + str(x)
    
rdd.foreach(showValues) # foreach (Applies a function to every element of rdd)
# useful to communicate to external services, accumulate values in a queue, logging info, ...
# NOTE: verify results in stderr file of the working dir

def showValuesPartition(iterator):
    vals = []
    for item in iterator:
        vals.append("hello: " + str(item))
    print vals
        
rdd.foreachPartition(showValuesPartition) # foreachPartition (Applies a function per partition of rdd)

rdd.max()

rdd.min()

rdd.stats()

rdd.sum()

rdd.mean()

rdd.stdev()

# must be an absolute path to directory name; default is hdfs namespace
# creates a part-xxxx file for each partition
rdd.saveAsTextFile("file:///" + os.path.realpath("./textfiles")) # saveAsTextFile

# using compression
# compresses part-xxxx file of each partition
rdd.saveAsTextFile("file:///" + os.path.realpath("./textfileszip"), 
                   compressionCodecClass = "org.apache.hadoop.io.compress.GzipCodec") # saveAsTextFile

rdd.saveAsPickleFile("file:///" + os.path.realpath("./textfiles-pickled")) # saveAsPickleFile (faster reads, writes)

rdd.countByValue() # countByValue - returns as dict of value: count

rdd.isEmpty() # isEmpty

print rdd.getStorageLevel() # getStorageLevel

rdd.getNumPartitions() # getNumPartitions

rdd.persist(StorageLevel.DISK_ONLY)
print rdd.is_cached
print rdd.getStorageLevel()
rdd.unpersist()
print rdd.is_cached
print rdd.getStorageLevel()

krdd = sc.parallelize([("a", 1), ("a", 2), ("b", 1), ("b", 2), ("c", 1)], 2)
krdd2 = sc.parallelize([("a", 3), ("b", 3), ("d", 1)], 2)

print krdd.glom().collect()
print krdd2.glom().collect()

krdd.groupByKey().collect() # groupByKey

list(krdd.groupByKey().collect()[0][1])

krdd.reduceByKey(lambda acc, x: acc + x, numPartitions = 1).collect() # reduceByKey
# does a groupByKey, followed by reduction
# operation must obey associative and commutative properties
# numPartitions controls the level of parallelism

# http://www.learnbymarketing.com/618/pyspark-rdd-basics-examples/
# does a groupByKey, followed by custom reduce function that doesn't have to obey commutative and associative property

# define a resultset template (any data structure) with initial values
init_state_template = [0]

def mergeValuesWithinPartition(template, val):
    template[0] = template[0] + val
    return template

def mergePartitions(template1, template2):
    template = template1[0] + template2[0]
    return template
    

krdd.aggregateByKey(init_state_template, 
                    mergeValuesWithinPartition, 
                    mergePartitions).collect() # aggregateByKey

krdd.sortByKey(ascending = False, numPartitions = 1, keyfunc = lambda x: x).collect() # sortByKey (can also use sortBy)

krdd.join(krdd2).collect() # join (inner-join in SQL; returns all-pair combinations)

krdd.leftOuterJoin(krdd2).collect() # leftOuterJoin (left join in SQL)

krdd.rightOuterJoin(krdd2).collect() # rightOuterJoin (right join in SQL)

krdd.fullOuterJoin(krdd2).collect() # fullOuterJoin (full join in SQL)

krdd.cogroup(krdd2).collect() # cogroup (returns iterator one for each rdd)

print list(krdd.cogroup(krdd2).collect()[0][1][0])
print list(krdd.cogroup(krdd2).collect()[0][1][1])

print list(krdd.cogroup(krdd2).collect()[2][1][0])
print list(krdd.cogroup(krdd2).collect()[2][1][1])

krdd.mapValues(lambda x: x**2).collect() # mapValues

krdd_val_iter = sc.parallelize([("a", [1,2,3]), ("b", [4,5,6])])
krdd_val_iter.flatMapValues(lambda x: [y**2 for y in x]).collect() # flatMapValues
# works in which value is an iterable object
# unpacks all elements in the iterable into their own key-value tuple/pair; puts them in a single list

krdd_val_iter.mapValues(lambda x: [y**2 for y in x]).collect() # mapValues 1-to-1 

krdd.keys().collect() # keys

krdd.values().collect() # values

krdd.collect()

krdd.count()

krdd.take(3)

krdd_dup = sc.parallelize([("a", 1), ("a", 1)]) 
krdd_dup.distinct().collect()

krdd.countByKey() # countByKey - number of times a key appears in the k-v rdd

krdd.lookup("a") # lookup

krdd.toDebugString() # toDebugString (identifies recursive dependencies of this rdd for debugging purposes)

krdd.collectAsMap() # collectAsMap -> return key-value RDD as a dictionary

# default accumulator accumulates only numeric (int and float) types; only does 'add' operation (commutative and associative)
accum = sc.accumulator(0, accum_param = None)

def squareValues(x):
    global accum
    #accum += 1
    accum.add(1)
    return x**2
    
print rdd.map(squareValues).collect()
print "No. of elements: %d" % accum.value

# custom accumulator to support any types
class CustomAccumulator(AccumulatorParam):
    
    def zero(self, initialValue):
        template = set()
        template.add(initialValue)
        return template
    
    def addInPlace(self, template1, template2):
        return template1.union(template2)
    
accum = sc.accumulator(None, accum_param = CustomAccumulator())

def squareValues(x):
    global accum
    accum += x
    return x**2
    
print rdd.map(squareValues).collect()
print "No. of elements: %d" % accum.value

bb = sc.broadcast({"a": 10, "b": 15})
print bb.value
bb.unpersist() # deletes cached copies from the executors

bb.value



