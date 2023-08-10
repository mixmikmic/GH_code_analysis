import requests

def set_hadoop_config(credentials):
    prefix = "fs.swift.service." + credentials['name'] 
    hconf = sc._jsc.hadoopConfiguration()
    hconf.set(prefix + ".auth.url", credentials['auth_url']+'/v2.0/tokens')
    hconf.set(prefix + ".auth.endpoint.prefix", "endpoints")
    hconf.set(prefix + ".tenant", credentials['project_id'])
    hconf.set(prefix + ".username", credentials['user_id'])
    hconf.set(prefix + ".password", credentials['password'])
    hconf.setInt(prefix + ".http.port", 8080)
    hconf.set(prefix + ".region", credentials['region'])
    hconf.setBoolean(prefix + ".public", True)

credentials = {
    'auth_url' : 'XXXXX',
    'project' : 'XXXXX',
    'project_id' : 'XXXXX',
    'region' : 'XXXXX',
    'user_id' : 'XXXXX',
    'domain_id' : 'XXXXX',
    'domain_name' : 'XXXXX',
    'username' : 'XXXXX',
    'password' : 'jXXXXX',
    'filename' : 'XXXX',
    'container' : 'XXXXX',
    'tenantId' : 'XXXXX'
}

bi_host = 'XXXXX'
bi_user = 'XXXXX'
bi_pass = 'XXXXX'
bi_folder = 'XXXXX' # destination folder in hdfs

credentials['name'] = 'keystone'
set_hadoop_config(credentials)
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

data = sc.wholeTextFiles("swift://notebooks." + credentials['name'] + "/*")

AUTH=(bi_user, bi_pass)

KEY=0
DATA=1

# FIXME! all files get read into memory - may crash with large/lots of files

for item in data.collect():
    filename = item[KEY].split('/')[-1]
    url = "{0}/webhdfs/v1/{1}/{2}?op=CREATE".format(bi_host, bi_folder, filename)
    
    print("started: {0} {1}".format(filename, url))
    
    # WARNING! certification verifcation is disabled as per the bluemix
    # documentation for curl with the -k flag
    
    response = requests.put(
        url, 
        auth = AUTH, 
        data = item[DATA].encode('utf-8'),
        verify = False,
        headers = { 'Content-Type' : 'text/plain; charset=utf8' }
    )
    
    if not response.status_code == requests.codes.ok:
        print(response.content)
    
    print('completed: ' + filename + '\n')





