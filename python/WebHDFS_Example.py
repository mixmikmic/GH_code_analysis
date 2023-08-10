#  Cluster number, e.g. 10000
cluster  = ''

# Cluster username
username = ''

# Cluster password
password = ''

# file path in HDFS
filepath = 'yourpath/yourfile.csv'

import pandas as pd
        
def read_csv_lines(lines, is_first_chunk = False):
    ''' returns a pandas dataframe '''
    
    if is_first_chunk:
        # you will want to set the header here if your datafile has a header record
        return pd.read_csv(lines, sep='|', header=None)
    else:
        return pd.read_csv(lines, sep='|', header=None)

host = 'ehaasp-{0}-mastermanager.bi.services.bluemix.net'.format(cluster)

import requests
import numpy as np
import sys
import datetime

if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO
    
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

print('SCRIPT START: {0}'.format(datetime.datetime.now()))

chunk_size = 10000000 # Read in 100 Mb chunks

url = "https://{0}:8443/gateway/default/webhdfs/v1/{1}?op=OPEN".format(host, filepath)

# note SSL verification is been disabled
r = requests.get(url, 
                 auth=(username, password), 
                 verify=False, 
                 allow_redirects=True, 
                 stream=True)

df = None
chunk_num = 1
remainder = ''
for chunk in r.iter_content(chunk_size):
    
    if chunk: # filter out keep-alive new chunks
        
        # Show progress by printing a dot - useful when chunk size is quite small
        # sys.stdout.write('.')
        # sys.stdout.flush()

        txt = remainder + chunk
        if '\n' in txt:
            [lines, remainder] = txt.rsplit('\n', 1)
        else:
            lines = txt

        if chunk_num == 1:
            pdf = read_csv_lines(StringIO(lines), True)
            df = sqlContext.createDataFrame(pdf)
        else:
            pdf = read_csv_lines(StringIO(lines), False)
            df2 = sqlContext.createDataFrame(pdf)
            
            df = df.sql_ctx.createDataFrame(
                    df._sc.union([df.rdd, df2.rdd]), df.schema
                    )
            
        print('Imported chunk: {0} record count: {1} df count: {2}'.format(chunk_num, len(pdf), df.count()))
            
        chunk_num = chunk_num + 1
        
print '\nTotal record import count: {0}'.format(df.count())

print('SCRIPT END: {0}'.format(datetime.datetime.now()))

df.cache()

df.show()



