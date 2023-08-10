import codecs
import json
import pprint

import pandas as pd

data = []
with codecs.open('../../data/SKNews.json', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        data.append(entry)

pprint.pprint(data[:2])
print(data[1]['body_text'])

pd_data = pd.DataFrame(data)

print(pd_data)

pd_data=pd_data.set_index('order')
print(pd_data.ix[0].body_text)

import os

# You need the pythia repo root on sys.path (or provided in the $PYTHONPATH env var)
from src.utils.load_json_to_pandas import load_json_as_pandas, load_json_file

json_path = '../../data/'
get_ipython().system('ls -al $json_path')

data_frame = load_json_as_pandas(json_path)

print(data_frame)

print(data_frame.sort_values('post_id'))

# Look to see how many documents are in each cluster
pd_data.groupby("cluster_id").size()

# how many clusters are there (those with non-zero which this current dataset has)?
len(pd_data.groupby("cluster_id").size())

from pyspark import SparkContext, SparkConf 
from pyspark.sql import SQLContext 

try:
    sc = SparkContext()
except:
    sc = SparkContext._active_spark_context

sqlCtx = SQLContext(sc)

data1 = sqlCtx.read.json("../../data/SKNews.json")

data1.collect()

data1.registerTempTable("stack")
sqlCtx.sql("select * from stack where order=0").take(1)

data1.printSchema()

