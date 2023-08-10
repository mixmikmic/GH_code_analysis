#!/usr/bin/python
#coding=utf-8

import pyspark
from pyspark import SparkConf, SparkContext
sc = None

conf = (SparkConf()
         .setMaster("local")
         .setAppName("MyApp")
         .set("spark.executor.memory", "1g")
         .set("packages","com.databricks:spark-xml_2.11:0.3.2"))
print(conf)

if sc is None:
    sc = SparkContext(conf = conf)
    
print(type(sc))
print(sc)

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

df = sqlContext.read.format('com.databricks.spark.xml').options(rowTag='book').load('books.xml')
df.select("author", "@id").write     .format('com.databricks.spark.xml')     .options(rowTag='book', rootTag='books')     .save('newbooks.xml')



