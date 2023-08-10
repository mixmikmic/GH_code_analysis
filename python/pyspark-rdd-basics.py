import pyspark

# Create a SparkContext in local mode
sc = pyspark.SparkContext("local")

# Parallelize a data set converting from an Array to an RDD
rdd = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Count the number of rows in the RDD
print rdd.count()

# View some rows
print rdd.take(10)

# Sort descending
descendingRdd = rdd.sortBy(lambda x: x, ascending = False)

# View some rows
print descendingRdd.take(10)

# Filter the RDD
filteredRdd = rdd.filter(lambda x: x < 5)

# View some rows
print filteredRdd.take(10)

# Map the RDD
rdd2 = rdd.map(lambda x: (x, x * 2))

# View some rows
print rdd2.take(10)

# Reduce the RDD by adding up all of the numbers
result = rdd.reduce(lambda a, b: a + b)

print result

# Load a Text file from HDFS
#textFile = sc.textFile("hdfs://...")

# Save an RDD to HDFS
#textFile.saveAsTextFile("hdfs://...")

# Parallelize a data set converting from an Array to an RDD
rdd = sc.parallelize(["aaa bbb ccc", "aaa bbb", "bbb ccc", "abc"])

# WordCount
results = rdd.flatMap(lambda line: line.split(" "))              .map(lambda word: (word, 1))              .reduceByKey(lambda a, b: a + b)

# Get the Results
results.take(10)

# Stop the context when you are done with it. When you stop the SparkContext resources 
# are released and no further operations can be performed within that context
sc.stop()

