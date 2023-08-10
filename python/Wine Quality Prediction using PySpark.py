import findspark
findspark.init('path_to_spark_installation_directory')

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

data = spark.read.csv('winequality-red.csv',
                      inferSchema = True, 
                      header = True,
                     sep = ';')

data.printSchema()

data.show()

data.describe().show()

data.columns

from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                           'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'],
                outputCol="features")
output = assembler.transform(data)
final_data = output.select('features', 'quality')
final_data.show()

# Splitting the data into trainind and test set
train, test = final_data.randomSplit([0.8, 0.2])

# Creating Regression model
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
lr = LinearRegression(labelCol = 'quality')
rfr = RandomForestRegressor(labelCol = 'quality', maxDepth = 10)

# Fit the model using training data
lrModel = lr.fit(train) 
rfrModel = rfr.fit(train)

# Evaluating Linear Regression Model

print('---------------- Linear Regression Model ----------------')
result_lr = lrModel.evaluate(test)
print("RMSE: {}".format(result_lr.rootMeanSquaredError))
print("MSE: {}".format(result_lr.meanSquaredError))
print("R2: {}".format(result_lr.r2))

# Predicting the quality of wine using linear regression model
prediction = lrModel.transform(test)
prediction.show(truncate = False)

# Predicting Random Forest Model
prediction_rfr = rfrModel.transform(test)
prediction_rfr.show()

# Evaluating Random Forest Model
from pyspark.ml.evaluation import RegressionEvaluator
rmse_eval = RegressionEvaluator(labelCol = 'quality', predictionCol = 'prediction', metricName = 'rmse')
mse_eval = RegressionEvaluator(labelCol = 'quality', predictionCol = 'prediction', metricName = 'mse')
r2_eval = RegressionEvaluator(labelCol = 'quality', predictionCol = 'prediction', metricName = 'r2')

rmse = rmse_eval.evaluate(prediction_rfr)
mse = mse_eval.evaluate(prediction_rfr)
r2 = r2_eval.evaluate(prediction_rfr)

print('---------------- Random Forest Model ----------------')
print("RMSE: {}".format(rmse))
print("MSE: {}".format(mse))
print("R2: {}".format(r2))



