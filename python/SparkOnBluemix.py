cluster  = '10451'    #  E.g. 10000
username = 'biadmin'  #  E.g. biadmin
password = ''         #  Please request password from chris.snow@uk.ibm.com
table    = 'biadmin.rowapplyout'  #  BigSQL table to query

import os
cwd = os.getcwd()

cls_host = 'ehaasp-{0}-mastermanager.bi.services.bluemix.net'.format(cluster)
sql_host = 'ehaasp-{0}-master-2.bi.services.bluemix.net'.format(cluster)

get_ipython().system('openssl s_client -showcerts -connect {cls_host}:9443 < /dev/null | openssl x509 -outform PEM > certificate')
    
# uncomment this for debugging
#!cat certificate 

get_ipython().system('rm -f truststore.jks')
get_ipython().system('keytool -import -trustcacerts -alias biginsights -file certificate -keystore truststore.jks -storepass mypassword -noprompt')

# test bigsql
url  = 'jdbc:db2://{0}:51000/bigsql:user={1};password={2};sslConnection=true;sslTrustStoreLocation={3}/truststore.jks;Password=mypassword;'.format(sql_host, username, password, cwd)
df = sqlContext.read.format('jdbc').options(url=url, driver='com.ibm.db2.jcc.DB2Driver', dbtable=table).load()

print(df.take(10))



