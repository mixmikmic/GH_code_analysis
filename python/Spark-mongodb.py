import os
import sys

#windows directory path containing spark binaries
spark_path = "C://opt//spark"

os.environ['SPARK_HOME'] = spark_path
os.environ['HADOOP_HOME'] = spark_path
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.mongodb.spark:mongo-spark-connector_2.10:2.0.0 pyspark-shell'

sys.path.append(spark_path + "//bin")
sys.path.append(spark_path + "//python")
sys.path.append(spark_path + "//python//pyspark//")
sys.path.append(spark_path + "//python//lib")
sys.path.append(spark_path + "//python//lib//pyspark.zip")
sys.path.append(spark_path + "//python//lib//py4j-0.10.4-src.zip")

from pyspark import SparkContext
sc = SparkContext(master="local[4]")

from pyspark.sql import SparkSession

my_spark = SparkSession     .builder     .appName("myApp")     .config("spark.mongodb.input.uri", "mongodb://127.0.0.1/people")     .config("spark.mongodb.output.uri", "mongodb://127.0.0.1/people")     .getOrCreate()

#write to mongodb
people = my_spark.createDataFrame([("Bilbo Baggins",  50), ("Gandalf", 1000), ("Thorin", 195), ("Balin", 178), ("Kili", 77),
   ("Dwalin", 169), ("Oin", 167), ("Gloin", 158), ("Fili", 82), ("Bombur", None)], ["name", "age"])
people.write.format("com.mongodb.spark.sql.DefaultSource").mode("append").option("database","local").option("collection", "people").save()

#read from mongodb
df = my_spark.read.format("com.mongodb.spark.sql.DefaultSource").option("uri","mongodb://127.0.0.1/local.people").load()

#print schema
df.printSchema()

df.collect()

type(df)

