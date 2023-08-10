import findspark
findspark.init()

import pyspark

conf = pyspark.conf.SparkConf()
(conf.setMaster('local[2]')
 .setAppName('ipython-notebook')
 .set("spark.executor.memory", "2g"))

sc = pyspark.SparkContext(conf=conf)

# distribute data into 3 slices
import time
rdd = sc.parallelize((time.time() for x in range(7)), numSlices=3)
print("Number of partitions: {}".format(rdd.getNumPartitions()))

print(rdd.glom().collect())

