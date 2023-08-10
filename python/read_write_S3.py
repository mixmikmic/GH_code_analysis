#Replace Accesskey with your amazon AccessKey and Secret with amazon secret
hconf = sc._jsc.hadoopConfiguration()  
hconf.set("fs.s3a.access.key", "XXXXXXXXXX")
hconf.set("fs.s3a.secret.key", "XXXXXXXXXX")

spark = SparkSession.builder.getOrCreate()
df_data_1 = spark.read  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')  .option('header', 'true')  .load('s3a://charlesbuckets31/FolderA/users.csv')
df_data_1.take(5)

df_data_1.printSchema()

df_data_1.write.save("s3a://charlesbuckets31/FolderB/users.parquet")

df_data_2 = spark.read  .format('org.apache.spark.sql.execution.datasources.parquet.ParquetFileFormat')  .option('header', 'true')  .load('s3a://charlesbuckets31/FolderB/users.parquet')
df_data_2.take(5)

