# We release the SparkContext if it exists.
try:
    sc
except:
    pass ;
else:
    sc.stop()

# Now handle initial import statements
from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext

# Create new Spark Configuration (port numbers might need to be adjusted from defaults.)
myconf = SparkConf()
myconf.setMaster('local[*]')
myconf.setAppName("INFO490 SP17 W14-NB1: Your Name")
myconf.set('spark.executor.memory', '1g')

# Create and initialize a new Spark Context
sc = SparkContext(conf=myconf)
ssc = StreamingContext(sc, 1)

# Display Spark version information, which also verifies SparkContext is active
print("\nSpark version: {0}".format(sc.version))

distFile = sc.textFile('./food_business_ids.pkl')

distFile.count()

# Create a DStream that will connect to hostname:port, like localhost:9090
lines = ssc.socketTextStream("localhost", 9090)

# Split each line into words
words = lines.flatMap(lambda line: line.split(" "))

# Count each word in each batch
pairs = words.map(lambda word: (word, 1))
wordCounts = pairs.reduceByKey(lambda x, y: x + y)

# Print the first ten elements of each RDD generated in this DStream to the console
wordCounts.pprint()

ssc.start()             # Start the computation
ssc.awaitTermination()  # Wait for the computation to terminate



