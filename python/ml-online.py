import json
import time
import pytz
import traceback
import time_uuid
from pytz import timezone
from datetime import datetime
from pyspark.sql import types
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import SQLContext, Row
from pyspark import SparkContext, SparkConf
from config import *

import warnings
import matplotlib
warnings.filterwarnings('ignore')

get_ipython().magic('matplotlib inline')
import pandas as pd

sc.stop()

conf = SparkConf()    .setAppName(APPNAME)    .setMaster(MASTER)    .set("spark.cassandra.connection.host", CASSANDRA_HOST)    .set("spark.cassandra.connection.port", CASSANDRA_PORT)    .set("spark.cassandra.auth.username", CASSANDRA_USERNAME)    .set("spark.cassandra.auth.password", CASSANDRA_PASSWORD)

sc = SparkContext(MASTER, APPNAME, conf=conf)
sqlContext = SQLContext(sc)
sqlContext.sql("""CREATE TEMPORARY TABLE %s                   USING org.apache.spark.sql.cassandra                   OPTIONS ( table "%s",                             keyspace "%s",                             cluster "Test Cluster",                             pushdown "true")               """ % (TABLE_QUERYABLE, TABLE_QUERYABLE, KEYSPACE))

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier as RF
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler, SQLTransformer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import numpy as np
import random
import functools
from pyspark.ml.feature import OneHotEncoder
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel

tableData = sqlContext.sql("SELECT * FROM %s" % (TABLE_QUERYABLE))
tableData.dtypes

cols_select = ['age', 'city', 'email', 'gender', 'job', 'zipcode']
df = tableData.select(cols_select).dropDuplicates()

def func(d):
    p = {}
    for x in d:
        p['age'] = d.age
        p['city'] = d.city
        p['email'] = d.email
        p['gender'] = d.gender
        p['job'] = d.job
        p['zipcode'] = d.zipcode
    p['behaviour'] = str(random.choice(['POSITIVE', 'NEGATIVE']))
    return p

features = df.map(lambda x: func(x))

from pyspark.sql.functions import UserDefinedFunction
def labelForAge(s):
    s = int(s)
    if s <= 20:
        return 0.0
    elif s > 20 and s <= 40:
        return 1.0
    elif s > 40 and s <= 50:
        return 2.0
    elif s > 50 and s <= 60:
        return 3.0
    else:
        return -1.0

def labelForCity(s):
    if len(s) <= 8:
        return 0.0
    elif s > 8 and s <= 10:
        return 1.0
    elif s > 10 and s <= 14:
        return 2.0
    elif s > 14 and s <= 20:
        return 3.0
    else:
        return -1.0

def labelForEmail(s):
    if len(s) <= 5:
        return 0.0
    elif s > 5 and s <= 7:
        return 1.0
    elif s > 7 and s <= 9:
        return 2.0
    elif s > 9 and s <= 12:
        return 3.0
    else:
        return -1.0

def labelForGender(s):
    if s == 'M':
        return 0.0
    elif s == 'F':
        return 1.0
    else:
        return -1.0
    
def labelForJob(s):
    s = s.lower()
    if 'engineer' in s:
        return 0.0
    elif 'architect' in s:
        return 1.0
    elif 'analyst' in s:
        return 2.0
    elif 'designer' in s:
        return 3.0
    elif 'officer' in s:
        return 4.0
    elif 'teacher' in s:
        return 5.0
    elif 'it' in s:
        return 6.0
    else:
        return -1.0   

def labelForZipcode(s):
    s = int(s)
    if s <= 10000:
        return 0.0
    elif s > 10000 and s <= 30000:
        return 1.0
    elif s > 30000 and s <= 50000:
        return 2.0
    elif s > 50000 and s <= 70000:
        return 3.0
    elif s > 70000 and s <= 90000:
        return 4.0
    elif s > 90000:
        return 5.0
    else:
        return -1.0

label_Age = UserDefinedFunction(labelForAge, DoubleType())
label_City = UserDefinedFunction(labelForCity, DoubleType())
label_Email = UserDefinedFunction(labelForEmail, DoubleType())
label_Gender = UserDefinedFunction(labelForGender, DoubleType())
label_Job = UserDefinedFunction(labelForJob, DoubleType())
label_Zipcode = UserDefinedFunction(labelForZipcode, DoubleType())

features_df = features.toDF()
labeledData = features_df.select(label_Age(features_df.age).alias('age_label'),                              label_City(features_df.city).alias('city_label'),                              label_Email(features_df.email).alias('email_label'),                              label_Gender(features_df.gender).alias('gender_label'),                              label_Job(features_df.job).alias('job_label'),                              label_Zipcode(features_df.zipcode).alias('zipcode_label'),                              features_df.behaviour)

cols_new = ['age_label', 'city_label', 'email_label', 'gender_label', 'job_label', 'zipcode_label']
assembler_features = VectorAssembler(inputCols=cols_new, outputCol='features')
labelIndexer = StringIndexer(inputCol='behaviour', outputCol="label")
tmp = [assembler_features, labelIndexer]
pipeline = Pipeline(stages=tmp)

allData = pipeline.fit(labeledData).transform(labeledData)
allData.cache()
print("Distribution of Pos and Neg in data is: ", allData.groupBy("label").count().take(3))

model = DecisionTreeModel.load(sc, "my_model.model")

model.toDebugString()

predictions = model.predict(allData.map(lambda x: x.features))
labelsAndPredictions = allData.map(lambda lp: lp.label).zip(predictions)

labelsAndPredictions.take(20)



