import pyspark

# Create a simple local Spark configuration.
conf = (
    pyspark
      .SparkConf()
      .setMaster('local[*]')
      .setAppName('Introduction Notebook')
)

# Show the configuration:
import pprint as pp
print('Configuration:')
pp.pprint(conf.getAll())

# Create a Spark context for local work.
try:
    sc
except:
    sc = pyspark.SparkContext(conf = conf)

# Check that we are using the expected version of PySpark.
print('Version: ',sc.version)

# Prove that Spark is installed and working correctly
rdd = sc.parallelize(range(1000))
result = rdd.takeSample(False, 5)
print('5 randomly selected values from the range: %s' % result)

