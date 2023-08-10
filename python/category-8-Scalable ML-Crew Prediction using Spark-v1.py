import findspark
findspark.init('C:\spark')
import pyspark

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('lr_crewprediction').getOrCreate()

from pyspark.ml.regression import LinearRegression

# Use Spark to read in the training data
ships = spark.read.csv("cruise_ship_train.csv",inferSchema=True,header=True)

# Print the Schema of the DataFrame
ships.printSchema()

ships.show()

ships.head()

for ship in ships.head(5):
    print(ship)
    print("\n")

ships.groupBy('Cruise_line').count().show()

ships.columns

from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol="Cruise_line", outputCol="CruiseLineIndex")
ships = indexer.fit(ships).transform(ships)
ships.show()

# Import VectorAssembler and Vectors
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=["Age", "Tonnage", "passengers",
               "length","cabins","passenger_density"],
    outputCol="numerical_features")

ships = assembler.transform(ships)
ships.show()

from pyspark.ml.feature import StandardScaler

scaler = StandardScaler(inputCol="numerical_features",
                        outputCol="scaled_numerical_features",
                        withStd=True, withMean=False)

# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(ships)

# Normalize each feature to have unit standard deviation.
ships = scalerModel.transform(ships)
ships.show()

# A few things we need to do before Spark can accept the data!
# It needs to be in the form of two columns
# ("label","features")
assembler = VectorAssembler(
    inputCols=["scaled_numerical_features","CruiseLineIndex"],
    outputCol="features")

shipsFinal = assembler.transform(ships)

shipsFinal.show()

final_data = shipsFinal.select("features",'crew')

train_data,test_data = final_data.randomSplit([0.8,0.2])

train_data.describe().show()

test_data.describe().show()

# Create a Linear Regression Model object
lr = LinearRegression(labelCol='crew')

# Fit the model to the data and call this model lrModel
lrModel = lr.fit(train_data)

# Print the coefficients and intercept for linear regression
print("Coefficients: {} Intercept: {}".format(lrModel.coefficients,lrModel.intercept))

test_results = lrModel.evaluate(test_data)

# Interesting results....
test_results.residuals.show()

print("RMSE: {}".format(test_results.rootMeanSquaredError))
print("MSE: {}".format(test_results.meanSquaredError))
print("RSquared: {}".format(test_results.r2))

# Use Spark to read in the testing data
ships_test = spark.read.csv("cruise_ship_test.csv",inferSchema=True,header=True)

indexer = StringIndexer(inputCol="Cruise_line", outputCol="CruiseLineIndex")
ships_test = indexer.fit(ships_test).transform(ships_test)
ships_test.show()

assembler = VectorAssembler(
    inputCols=["Age", "Tonnage", "passengers",
               "length","cabins","passenger_density"],
    outputCol="numerical_features")

ships_test = assembler.transform(ships_test)
ships_test.show()

scaler = StandardScaler(inputCol="numerical_features",
                        outputCol="scaled_numerical_features",
                        withStd=True, withMean=False)

# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(ships_test)

# Normalize each feature to have unit standard deviation.
ships_test = scalerModel.transform(ships_test)
ships_test.show()

# A few things we need to do before Spark can accept the data!
# It needs to be in the form of two columns
# ("label","features")
assembler = VectorAssembler(
    inputCols=["scaled_numerical_features","CruiseLineIndex"],
    outputCol="features")

ships_test_Final = assembler.transform(ships_test)

ships_test_Final.show()

final_data_test = ships_test_Final.select("features")

predictions = lrModel.transform(final_data_test)

predictions.show()



