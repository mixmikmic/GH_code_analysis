get_ipython().magic('Addjar -f https://jdbc.postgresql.org/download/postgresql-9.4.1207.jre7.jar')

#Now change the kernel back to Python

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

#Ignore the Connection Error because of the invalid connection details
#Just simply change the publichost to your hostname and port number and databasename and
#tablename

df = sqlContext.load(source="jdbc",                 url="jdbc:postgresql://[publichost]:[port]/databasename",                 dbtable="[tablename]")



