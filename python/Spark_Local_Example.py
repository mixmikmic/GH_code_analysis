import os
import sys

# Note SPARK_HOME is set in my .bashrc file
#Set PYSPARK_PYTHON to the path of the local interpreter or environment
os.environ["PYSPARK_PYTHON"]="/Users/johnpatanian/anaconda/bin/python" 
sys.path.insert(0, os.environ["PYLIB"] +"/py4j-0.9-src.zip")
sys.path.insert(0, os.environ["PYLIB"] +"/pyspark.zip")

import re

import pandas as pd
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, functions as F
from pyspark.sql.functions import array
from pyspark.mllib.stat import Statistics
from pyspark.mllib.clustering import KMeans, KMeansModel

conf = SparkConf()
conf.setMaster('local')
conf.setAppName('phm_demo')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

#Tag Names provided in data set description
tag_names = ['unit','cycle_num', 'setting1','setting2',
         'setting3', 'sensor1', 'sensor2',
         'sensor3', 'sensor4', 'sensor5', 'sensor6',
         'sensor7', 'sensor8', 'sensor9', 'sensor10',
         'sensor11', 'sensor12', 'sensor13', 'sensor14',
         'sensor15', 'sensor16', 'sensor17', 'sensor18',
         'sensor19', 'sensor20', 'sensor21']

train_data = pd.read_csv('/Users/johnpatanian/Documents/PHM_2016/PHM2016/train.txt', 
                         sep='\s+', header=None, names=tag_names)

# Create a Spark DataFrame from a pandas DataFrame
spark_train_data = sqlContext.createDataFrame(train_data)

spark_train_data[['sensor1', 'sensor2', 'sensor3']].show(5)

spark_train_data[['setting1', 'setting2', 'setting3']].describe().show()

def get_sensor_names(tag_names):
    """ Get tagnames starting with sensor.
    
    :param tag_names: Input time series data frame
    
    :return list of string tag names starting with sensor.
    """
    
    return [tag_name for tag_name in tag_names if re.search('^sensor.',tag_name)]

# Note I am using the Spark DataFrame's column property here.
sensor_columns = get_sensor_names(spark_train_data.columns)

# Now look at the top 5 rows of the first 8 columns the sensor data
spark_train_data[sensor_columns[0:8]].show(5)

import numpy as np

def convert_df_to_rdd(input_df):
    """ Convert to an rdd and then convert each row to a numpy array. """
    return input_df.rdd.map(lambda row: np.array(row))
    
settings_columns = ['setting1', 'setting2', 'setting3']
spark_train_rdd = convert_df_to_rdd(spark_train_data[sensor_columns])

spark_train_settings_rdd = convert_df_to_rdd(spark_train_data[settings_columns])

from pyspark.sql import Row
from pyspark.sql.types import IntegerType

clusters = KMeans.train(spark_train_settings_rdd, 6, maxIterations=10000, 
                        seed=0, initializationSteps=20)

centers = clusters.centers
centers

column_list = spark_train_data.columns
column_list.insert(len(column_list), 'overall_setting')

spark_train_data = spark_train_data.map(lambda row: row + 
                     Row(overall_setting=clusters.predict(np.array(row[2:5])))).toDF(column_list)

spark_train_data = spark_train_data.drop('setting1')
spark_train_data = spark_train_data.drop('setting2')
spark_train_data = spark_train_data.drop('setting3')

spark_train_data.groupBy('overall_setting').count().show()

aggregates = {sensor: "variance" for sensor in sensor_columns}

sens_var = spark_train_data.groupby('overall_setting').agg(aggregates)


sens_var[['overall_setting', 'variance(sensor1)', 'variance(sensor2)', 
          'variance(sensor3)', 'variance(sensor4)','variance(sensor5)']].show()

sens_var[['overall_setting', 'variance(sensor6)', 'variance(sensor7)', 
          'variance(sensor8)', 'variance(sensor9)']].show()

sens_var[['overall_setting', 'variance(sensor10)', 'variance(sensor11)', 
          'variance(sensor12)', 'variance(sensor13)']].show()

sens_var[['overall_setting', 'variance(sensor14)', 'variance(sensor15)', 
          'variance(sensor16)', 'variance(sensor17)']].show()

sens_var[['overall_setting', 'variance(sensor18)', 'variance(sensor19)', 
          'variance(sensor20)', 'variance(sensor21)']].show()

### Drop the Columns with Zero or Near Zero Variance
drop_columns = ['sensor1','sensor5','sensor18','sensor19','sensor8',
                'sensor11','sensor13','sensor15','sensor20','sensor21']

for column in drop_columns:
    spark_train_data = spark_train_data.drop(column)

spark_train_data.show(5)

sensor_columns = get_sensor_names(spark_train_data.columns)
spark_train_rdd = convert_df_to_rdd(spark_train_data[sensor_columns])

print Statistics.corr(spark_train_rdd, method="pearson")



