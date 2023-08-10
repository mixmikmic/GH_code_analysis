import findspark
findspark.init('Path_to_Spark_Installation')

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

data = spark.read.csv('hack_data.csv',
                      header = True,
                     inferSchema = True)
data.printSchema()

data.head(1)

data.columns

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols = ['Session_Connection_Time', 'Bytes Transferred', 'Kali_Trace_Used',
                                         'Servers_Corrupted', 'Pages_Corrupted', 'WPM_Typing_Speed'],
                           outputCol = 'features')
final_data = assembler.transform(data)

from pyspark.ml.feature import StandardScaler

scaler = StandardScaler(inputCol = 'features', 
                        outputCol = 'scaledFeatures', 
                        withStd = True, 
                        withMean = True)
scaled_data = scaler.fit(final_data).transform(final_data)

# Let's check with K = 2 and K =3. Since that is what we suspect

from pyspark.ml.clustering import KMeans

kmeans_2 = KMeans(k = 2, featuresCol = 'scaledFeatures')
kmeans_3 = KMeans(k = 3, featuresCol = 'scaledFeatures')

# Fitting the KMeans model with the data
model_2 = kmeans_2.fit(scaled_data)
model_3 = kmeans_3.fit(scaled_data)

# Evaluating the performance of the model(Within Set Sum of Squared Error)
print('WSSE with K = 2')
print(model_2.computeCost(scaled_data))
print('--'*15)
print('WSSE with K = 3')
print(model_3.computeCost(scaled_data))

# to check which point belongs to which cluster(because we can not visualize higher-dimension data)
results_2 = model_2.transform(scaled_data) 
print('Clustering with K = 2')
results_2.drop('features', 'scaledFeatures', 'Location').show()

results_3 = model_3.transform(scaled_data) 
print('Clustering with K = 3')
results_3.drop('features', 'scaledFeatures', 'Location').show()

# Let's count the number of record that belongs to each hacker

print('Number of hacks by each hacker when K = 2')
results_2.groupBy('prediction').count().show()

print('\nNumber of hacks by each hacker when K = 3')
results_3.groupBy('prediction').count().show()



