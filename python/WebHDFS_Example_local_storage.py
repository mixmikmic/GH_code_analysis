#  Cluster number, e.g. 100000
cluster  = ''

# Cluster username
username = ''

# Cluster password
password = ''

# file path in HDFS
webhdfs_filepath = 'yourpath/yourfile.txt'

# where to save the file in the spark service file system
local_filepath = 'yourfile.txt'

host = 'ehaasp-{0}-mastermanager.bi.services.bluemix.net'.format(cluster)

import requests
import sys
import datetime
    
print('READ FILE START: {0}'.format(datetime.datetime.now()))

chunk_size = 200000000 # Read in 200 Mb chunks

url = "https://{0}:8443/gateway/default/webhdfs/v1/{1}?op=OPEN".format(host, webhdfs_filepath)

# note SSL verification is been disabled
r = requests.get(url, 
                 auth=(username, password), 
                 verify=False, 
                 allow_redirects=True, 
                 stream=True)
chunk_num = 1
with open(local_filepath, 'wb') as f:
    for chunk in r.iter_content(chunk_size):
        if chunk: # filter out keep-alive new chunks
           print('{0} writing chunk {1}'.format(datetime.datetime.now(), chunk_num))
           f.write(chunk)
           chunk_num = chunk_num + 1
        
print('READ FILE END: {0}'.format(datetime.datetime.now()))

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

df = sqlContext.read.format('com.databricks.spark.csv')             .options(header='false', inferschema='true', delimiter='|')             .load(local_filepath)
df.cache()
df.show()

df.count()



