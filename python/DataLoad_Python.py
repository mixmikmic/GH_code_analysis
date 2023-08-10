import getpass

#insert this into password's value in dictionary
password = getpass.getpass()
#display that it works
print password



import requests, StringIO, json

def get_file_content(credentials):
    """For given credentials, this functions returns a StringIO object containg the file content 
    from the associated Bluemix Object Storage V3."""

    url1 = ''.join([credentials['auth_url'], '/v3/auth/tokens'])
    data = {'auth': {'identity': {'methods': ['password'],
                                  'password': {'user': {'name': credentials['username'],
                                                        'domain': {'id': credentials['domain_id']},
                                                        'password': credentials['password']}}}}}
    headers1 = {'Content-Type': 'application/json'}
    resp1 = requests.post(url=url1, data=json.dumps(data), headers=headers1)
    resp1_body = resp1.json()
    for e1 in resp1_body['token']['catalog']:
        if(e1['type']=='object-store'):
            for e2 in e1['endpoints']:
                if(e2['interface']=='public'and e2['region']==credentials['region']):
                    url2 = ''.join([e2['url'],'/', credentials['container'], '/', credentials['filename']])
                    print url2

    s_subject_token = resp1.headers['x-subject-token']
    headers2 = {'X-Auth-Token': s_subject_token, 'accept': 'application/json'}
    resp2 = requests.get(url=url2, headers=headers2)
    return StringIO.StringIO(resp2.content)

import pandas as pd

data_df = pd.read_csv(get_file_content(credentials))
data_df.head()

def set_hadoop_config(credentials):
    """This function sets the Hadoop configuration with given credentials, 
    so it is possible to access data using SparkContext"""
    
    prefix = "fs.swift.service." + credentials['name']
    hconf = sc._jsc.hadoopConfiguration()
    hconf.set(prefix + ".auth.url", credentials['auth_url']+'/v3/auth/tokens')
    hconf.set(prefix + ".auth.endpoint.prefix", "endpoints")
    hconf.set(prefix + ".tenant", credentials['project_id'])
    hconf.set(prefix + ".username", credentials['user_id'])
    hconf.set(prefix + ".password", credentials['password'])
    hconf.setInt(prefix + ".http.port", 8080)
    hconf.set(prefix + ".region", credentials['region'])
    hconf.setBoolean(prefix + ".public", True)

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

# you can choose any alphanumeric name
credentials['name'] = 'keystone'
set_hadoop_config(credentials)

data_rdd = sc.textFile("swift://" + credentials['container'] + "." + credentials['name'] + "/" + credentials['filename'])
data_rdd.take(5)

def set_hadoop_config(name, credentials):
    prefix = "fs.swift2d.service." + name 
    hconf = sc._jsc.hadoopConfiguration()
    hconf.set("fs.swift2d.impl", "com.ibm.stocator.fs.ObjectStoreFileSystem")
    hconf.set(prefix + ".auth.url", credentials['auth_url']+'/v3/auth/tokens')
    hconf.set(prefix + ".auth.endpoint.prefix", "endpoints")
    hconf.set(prefix + ".tenant", credentials['project_id'])
    hconf.set(prefix + ".username", credentials['user_id'])
    hconf.set(prefix + ".password", credentials['password'])
    hconf.setInt(prefix + ".http.port", 8080)
    hconf.set(prefix + ".region", credentials['region'])
    hconf.setBoolean(prefix + ".public", True)

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

set_hadoop_config("sparksql", credentials)
print sc._jsc.hadoopConfiguration().get("fs.swift2d.impl")

FILENAME = credentials["filename"]
FILENAME2D = "swift2d://notebooks.sparksql/" + FILENAME

data_rdd = sc.textFile(FILENAME2D)
data_rdd.take(5)



from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

TABLENAME = "<name>"

props = {}
props['user'] = credentials_2['username']
props['password'] = credentials_2['password']

table = credentials_2['username'] + "." + TABLENAME

data_df = sqlContext.read.jdbc(credentials_2['jdbcurl'],table,properties=props)
data_df.printSchema()

data_df.take(5)



get_ipython().system('pip install --user cloudant')

from cloudant.client import Cloudant
from cloudant.result import Result
import pandas as pd, json

client = Cloudant(credentials_3['username'], credentials_3['password'], url=credentials_3['url'])
client.connect()

client.all_dbs()

# fill in database name 
db_name = 'test_db'
my_database = client[db_name]
result_collection = Result(my_database.all_docs, include_docs=True)
data_df = pd.DataFrame([item['doc'] for item in result_collection])
data_df.head()

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

# fill in database name 
db_name = "test_db"
data_df = sqlContext.read.format("com.cloudant.spark").option("cloudant.host",credentials_3['host']).option("cloudant.username",credentials_3['username']).option("cloudant.password",credentials_3['password']).load(db_name)

data_df.printSchema()

data_df.take(5)

get_ipython().system('pip install --user pymongo')

import ssl
import json
import pymongo
from pymongo import MongoClient

host = '<host>'
database = '<db>'
collection = '<col>'

client = MongoClient(host+database, ssl=True, ssl_cert_reqs=ssl.CERT_NONE)
db = client[database]
col = db[collection]

cursor = col.find()
json_content = []
json_file = 'mongo.json'

for doc in cursor:
    doc['_id'] = str(doc['_id'])
    json_content.append(doc)
data = json.dumps(json_content)

with open(json_file, 'w') as text_file:
      text_file.write(data)

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

df = sqlContext.read.json(json_file)
df.printSchema()
df.show(5)

# !rm <json_file>

#%Addjar https://jdbc.postgresql.org/download/postgresql-9.4.1208.jre6.jar 

host = '<host>'
port = '<port>'
user = '<user>'
password = '<password>'
dbname = '<db>'
dbtable = '<table>'

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

df = sqlContext.read.format('jdbc')                    .options(url='jdbc:postgresql://'+host+':'+port+'/'+dbname+'?user='+user+'&password='+password, dbtable=dbtable)                    .load()
df.printSchema()
df.show(5)

