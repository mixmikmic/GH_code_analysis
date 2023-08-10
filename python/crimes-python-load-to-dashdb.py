# Import and initialize SparkSession
from pyspark.sql import SparkSession
spark = SparkSession    .builder    .config('cloudant.host', 'examples.cloudant.com')    .getOrCreate()

# Load Cloudant documents from 'crimes' into Spark DataFrame
cloudantdata = spark.read.format('org.apache.bahir.cloudant').load('crimes')

# In case of doing multiple operations on a dataframe (select, filter etc.)
# you should persist the dataframe.
# Othewise, every operation on the dataframe will load the same data from Cloudant again.
# Persisting will also speed up computation.
cloudantdata.cache() # persisting in memory

# Print the schema
cloudantdata.printSchema()

propertiesDF = cloudantdata.select('properties').withColumn('properties.compnos', cloudantdata['properties.compnos']).withColumn('properties.domestic', cloudantdata['properties.domestic']).withColumn('properties.fromdate', cloudantdata['properties.fromdate']).withColumn('properties.main_crimecode', cloudantdata['properties.main_crimecode']).withColumn('properties.naturecode', cloudantdata['properties.naturecode']).withColumn('properties.reptdistrict', cloudantdata['properties.reptdistrict']).withColumn('properties.shooting', cloudantdata['properties.shooting']).withColumn('properties.source', cloudantdata['properties.source']).drop('properties')

propertiesDF.printSchema()

import pixiedust

get_ipython().run_cell_magic('scala', 'cl=dialect global=true', 'import org.apache.spark.sql.jdbc._\nimport org.apache.spark.sql.types.{StringType, BooleanType, DataType}\n\nobject db2CustomDialect extends JdbcDialect {\n    override def canHandle(url: String): Boolean = url.startsWith("jdbc:db2")\n    override def getJDBCType(dt: DataType): Option[JdbcType] = dt match {\n            case StringType => Option(JdbcType("VARCHAR(50)", java.sql.Types.VARCHAR))\n            case BooleanType => Option(JdbcType("CHAR(1)", java.sql.Types.CHAR))\n            case _ => None\n    }\n}\nJdbcDialects.registerDialect(db2CustomDialect)')

conn_properties = {
   'user': 'username',
   'password': 'password',
   'driver': 'com.ibm.db2.jcc.DB2Driver'
}

db2_jdbc_url = 'jdbc:db2://***:50000/BLUDB'

# Save Spark DataFrame to Db2 Warehouse
propertiesDF.write.jdbc(db2_jdbc_url, 'crimes_filtered', properties=conn_properties)

