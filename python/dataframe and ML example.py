

get_ipython().system('wget https://raw.githubusercontent.com/daniel-acuna/hackathon_syracuse/master/data/RoadRatings2015.csv')

get_ipython().system('wget https://raw.githubusercontent.com/daniel-acuna/hackathon_syracuse/master/data/potholes.csv')

get_ipython().system('pwd')

roadratings_df = sqlContext.read    .format("com.databricks.spark.csv")    .option("header", "true")    .option("inferSchema", "true")    .option("nullValue", 'NA')    .load("file:///home/ischool/spark_demo/RoadRatings2015.csv")

potholes_df = sqlContext.read    .format("com.databricks.spark.csv")    .option("header", "true")    .option("inferSchema", "true")    .option('nullValue', 'NA')    .load("file:///home/ischool/spark_demo/potholes.csv")

from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

rr_df = roadratings_df.select(['streetID', 'overall', 'crack', 'patch', 'length', 'width'])

ph_df = potholes_df.select(['STREET_ID', 'Latitude', 'Longitude'])

from pyspark.sql import functions

ph_df.groupBy('STREET_ID').agg(functions.count('*').alias('n_potholes')).orderBy(functions.desc('n_potholes')).show()

## Your code here

ph_df.groupBy('STREET_ID').agg(functions.count('*').alias('n_potholes')).orderBy(functions.desc('n_potholes')).    join(potholes_df.select(['STREET_ID', 'strLocation']), 'STREET_ID').show()

rr_df.columns

dataset_df = rr_df.join(ph_df, rr_df['streetID'] == ph_df['STREET_ID'])

dataset_df.printSchema()

lr = LinearRegression(featuresCol='features', labelCol='overall')

va = VectorAssembler(inputCols=['length'], outputCol='features')

pl = Pipeline(stages=[va, lr])

input_dataset_df = dataset_df.select(dataset_df['length'].astype('double').alias('length'),
                         dataset_df['overall'].astype('double').alias('overall')
                        ).dropna()

training_df, testing_df = input_dataset_df.randomSplit([0.8, 0.2])

training_df.count()

testing_df.count()

pl_fit = pl.fit(training_df)

pl_fit.transform(training_df).show()

training_df.count()

testing_df.count()

pl_fit.transform(testing_df).show()





