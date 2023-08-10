from datetime import datetime

from pyspark import SparkContext, SQLContext
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, TimestampType, DoubleType, StringType

from sparkts.datetimeindex import uniform, BusinessDayFrequency
from sparkts.timeseriesrdd import time_series_rdd_from_observations

def loadObservations(sparkContext, sqlContext, path):
    textFile = sparkContext.textFile(path)
    rowRdd = textFile.map(lineToRow)
    schema = StructType([
        StructField('timestamp', TimestampType(), nullable=True),
        StructField('symbol', StringType(), nullable=True),
        StructField('price', DoubleType(), nullable=True),
    ])
    return sqlContext.createDataFrame(rowRdd, schema);

def lineToRow(line):
    (year, month, day, symbol, volume, price) = line.split("\t")
    # Python 2.x compatible timestamp generation
    dt = datetime(int(year), int(month), int(day))
    return (dt, symbol, float(price))

#sc = SparkContext(appName="Stocks")
sqlContext = SQLContext(sc)

data_path = "/Users/sundeepblue/spark-ts-examples-master/data/ticker.tsv"
tickerObs = loadObservations(sc, sqlContext, data_path)

tickerObs

# Note:

# It appeared that the sample code (https://github.com/sryza/spark-ts-examples/blob/master/python/Stocks.py) 
# provided by spark-ts does not work. I suspect there is still a bug in the function python 
# class 'BusinessDayFrequency', but the fix made by the spark-ts team was not yet pushed to the repository

# See related posts by others who encountered the same issue:
# https://groups.google.com/forum/#!topic/spark-ts/Nl2w2Sq1VIY
# https://forums.databricks.com/questions/6575/how-can-i-use-spark-ts-spark-time-series-library-b.html

# Create an daily DateTimeIndex over August and September 2015

freq = BusinessDayFrequency(1, 1, sc) # ERROR will occur here showing "JavaPackage object is not callable"
dtIndex = uniform(start='2015-08-03T00:00-07:00', end='2015-09-22T00:00-07:00', freq=freq, sc=sc)

# The functionality of spark-ts is still very limited to process/predict stock price
# https://github.com/sryza/spark-ts-examples
# therefore I will use a python library called "statsmodels" to manipulate stock/S&P index.

# see another jupyter notebook

