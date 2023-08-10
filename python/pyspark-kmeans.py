# Import the needed libraries
import pyspark
from pyspark.mllib.linalg import Vectors
from pyspark.ml.clustering import KMeans

# Create a SparkContext in local mode
sc = pyspark.SparkContext("local")

# Create a SqlContext from the SparkContext
sqlContext = pyspark.SQLContext(sc)

# Setup data for some East Coast and West Coast cities
data = [ 
    # City, Latitude, Longitude
    ( 'San Francisco,CA', Vectors.dense(37.62, 122.38) ),
    ( 'San Jose,CA', Vectors.dense(37.37, 121.92) ),
    ( 'Portland,OR', Vectors.dense(45.60, 122.60) ),
    ( 'Seattle,WA', Vectors.dense(47.45, 122.30) ),
    ( 'New York,NY', Vectors.dense(40.77, 73.98) ),
    ( 'Atlantic City,NJ', Vectors.dense(39.45, 74.57) ),
    ( 'Philadelphia,PA', Vectors.dense(39.88, 75.25) ),
    ( 'Boston,MA', Vectors.dense(42.37, 71.03) ),
    ( 'Santa Rosa,CA', Vectors.dense(38.52, 122.82) )
]

# Create a DataFrame
df = sqlContext.createDataFrame(data, ['city', 'features'])

# Convert to a Pandas DataFrame for easy display
df.toPandas()

# Setup KMeans, where k is the number of cluster centers we want
kmeans = KMeans(k=2, seed=1)

# Train the model
model = kmeans.fit(df)

# Print the cluster centers
print "cluster centers: " + str(model.clusterCenters())

# Use the model to cluster the original frame
results = model.transform(df).select("city", "features", "prediction")

# Convert results to Pandas DataFrame for easy display
results.toPandas()

# Setup KMeans, where k is the number of cluster centers we want
kmeans = KMeans(k=3, seed=1)

# Train the Model
model = kmeans.fit(df)

# Use the model to cluster the original frame
results = model.transform(df).select("city", "features", "prediction")

# Convert results to Pandas DataFrame for easy display
results.toPandas()

# Stop the context when you are done with it. When you stop the SparkContext resources 
# are released and no further operations can be performed within that context
sc.stop()

