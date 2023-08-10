get_ipython().system('wget https://dl.dropboxusercontent.com/s/zegtlp7q47qltdh/Chinook_Sqlite.sqlite')

get_ipython().system('ls')

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

df = sqlContext.read.format('jdbc').     options(url='jdbc:sqlite:Chinook_Sqlite.sqlite',     dbtable='employee',driver='org.sqlite.JDBC').load()

df.printSchema()



