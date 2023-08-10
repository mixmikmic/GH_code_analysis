get_ipython().run_cell_magic('bash', '', 'time spark-submit --version')

get_ipython().run_cell_magic('time', '', 'from pyspark.sql import SparkSession')

get_ipython().run_cell_magic('time', '', 'spark = (SparkSession.builder.appName("test").getOrCreate())')

spark.version

get_ipython().run_cell_magic('bash', '', 'wget https://www.sharcnet.ca/~jnandez/power_data.csv')

get_ipython().run_cell_magic('time', '', 'dataDF = spark.read.csv("power_data.csv",header=True)')

dataDF.printSchema()

get_ipython().run_cell_magic('time', '', 'dataDF = spark.read.csv("power_data.csv",header=True,inferSchema=True)')

dataDF.printSchema()

get_ipython().run_cell_magic('bash', '', 'squeue -u $USER')



