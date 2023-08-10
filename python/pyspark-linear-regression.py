# Import the needed libraries
import pyspark
from pyspark.mllib.linalg import Vectors
from pyspark.ml.regression import LinearRegression

# Create a SparkContext in local mode
sc = pyspark.SparkContext("local")

# Create a SqlContext from the SparkContext
sqlContext = pyspark.SQLContext(sc)

# Setup some fictional housing data
data = [ 
    # price, sqft, bedrooms
    ( 300000.0, Vectors.dense( 2000.0, 3.0 ) ),
    ( 500000.0, Vectors.dense( 4000.0, 4.0 ) ),
    ( 250000.0, Vectors.dense( 1500.0, 2.0 ) ),
    ( 165000.0, Vectors.dense( 1200.0, 1.0 ) ),
    ( 325000.0, Vectors.dense( 2500.0, 3.0 ) ),
    ( 275000.0, Vectors.dense( 1900.0, 3.0 ) ) 
]

# Create a DataFrame
df = sqlContext.createDataFrame(data, ['price', 'features'])

# Convert to a Pandas DataFrame for easy display
df.toPandas()

# Setup LinearRegression
lr = LinearRegression(maxIter=5, regParam=0.0, labelCol="price")

# Train the model
model = lr.fit(df)

# View properties of the trained model
print "intercept: " + str(model.intercept)
print "weights: " + str(model.weights)

# Setup data that we want to do predictions for
dataToPredict = [
    # sqft, bedrooms
    ( Vectors.dense(2700.0, 3.0), ),
    ( Vectors.dense(1700.0, 3.0), ),
    ( Vectors.dense(1700.0, 2.0), ),
    ( Vectors.dense(1000.0, 1.0), )
]

# Create a DataFrame
dfToPredict = sqlContext.createDataFrame(dataToPredict, ["features"])
     
# Use the model to predict housing prices
predictions = model.transform(dfToPredict)

# Convert to a Pandas DataFrame for easy display
predictions.toPandas()

# Stop the context when you are done with it. When you stop the SparkContext resources 
# are released and no further operations can be performed within that context
sc.stop()

