print sc
print sqlContext
print sqlCtx

bike_df = (sqlContext
           .read
           .format('com.databricks.spark.csv')
           .option("header", "true") # Use first line of all files as header
           .option("inferSchema", "true") # Automatically infer data types
           .load("bike-data/day.csv"))

bike_df.columns

bike_df.show()

bike_df1 = bike_df.select('season','mnth','holiday','weekday','workingday','weathersit','temp','atemp','hum',
                         'windspeed','casual','registered','cnt')

bike_df2 = bike_df1.withColumn("cnt", bike_df1["cnt"].cast("double"))

bike_df2.show()

bike_df2.dtypes

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder,CrossValidator

feature_columns = ['season'
                   ,'mnth'
                   ,'holiday'
                   ,'weekday'
                   ,'workingday'
                   ,'weathersit'
                   ,'temp'
                   ,'atemp'
                   ,'hum'
                   ,'windspeed'
                   ,'casual'
                   ,'registered']

(training_data, test_data) = bike_df2.randomSplit([0.7,0.3], seed = 10)
print "Training data size is :"+str(training_data.count())
print "Test data size is :"+str(test_data.count())

training_data.dtypes

vecAssembler = VectorAssembler(inputCols=feature_columns, outputCol='features_vector')
#bike_df2 = vecAssembler.transform(bike_df1)

rdf = RandomForestRegressor(labelCol='cnt',featuresCol="features_vector",predictionCol='predicted_cnt',seed=15)

pipeline = Pipeline(stages=[vecAssembler,rdf])

paramGrid = (ParamGridBuilder()
             .addGrid(rdf.maxDepth,[5,10,15,20])
             .addGrid(rdf.numTrees,[1,10,50,100])
             .build())

rdfEvaluator = RegressionEvaluator(predictionCol="predicted_cnt", labelCol='cnt', metricName='rmse')

cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=rdfEvaluator)

# Run cross-validation, and choose the best set of parameters.
cvModel = cv.fit(training_data)

test_data_with_predictions = cvModel.transform(test_data)

test_data_with_predictions.show()

test_data_RMSE = rdfEvaluator.evaluate(test_data_with_predictions)
print "RMSE on test data is : " + str(test_data_RMSE)

bestRDFModel = cvModel.bestModel.stages[1]

bestRDFModel



