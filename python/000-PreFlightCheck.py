import datetime
from pytz import timezone
print "Last run @%s" % (datetime.datetime.now(timezone('US/Pacific')))
#
import sys
print "Python Version : %s" % (sys.version)
#
from pyspark.sql import SparkSession 
print "Spark Version  : %s" % (spark.version)

from pyspark.context import SparkContext
print "Spark Version  : %s" % (sc.version)
#
from pyspark.conf import SparkConf
conf = SparkConf()
print conf.toDebugString()

data = xrange(1,101)

data_rdd = sc.parallelize(data)

data_rdd.take(3)

# Make sure rdd works
data_rdd.filter(lambda x: x < 10).collect()

data_rdd.top(5)

# Move on to dataFrames
df = data_rdd.map(lambda x:[x]).toDF(['SeqNo']) # needs each row as a list

df.show(10)

df.filter(df.SeqNo <= 10).show()

import pyspark.sql.functions as F
df.withColumn("Square",F.pow(df.SeqNo,2)).show(10) # Normal pow doesn't take columns

# Reduce vs fold
rdd_1 = sc.parallelize([])

rdd_1.reduce(lambda a,b : a+b)

rdd_1.take(10)

rdd_1.fold(0,lambda a,b : a+b)

rdd_2 = sc.parallelize([1,2])

from operator import add
rdd_2.fold(0,add)

rdd_x = sc.parallelize(['a','b','c'])
rdd_y = sc.parallelize([1,2,3])

rdd_x.cartesian(rdd_y).take(20)



