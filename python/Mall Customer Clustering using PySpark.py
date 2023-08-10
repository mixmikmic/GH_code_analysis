# Supressing all the warnings
import warnings
warnings.filterwarnings('ignore')

import findspark
findspark.init('Path_to_Spark_Installation_Folder')

# Initializing the Spark Session
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# Reading the data
data = spark.read.csv('Customers.csv', header = True, inferSchema = True)

data.printSchema()

data.show(10)

# Considering only two columns(Annual Income (k$) and Spending Score (1-100)) so that we can visualize our data in 2D
# In real world pronem we want to as many significant features as available
from pyspark.sql.functions import *
data = data.select(col('Annual Income (k$)').alias('Income'), col('Spending Score (1-100)').alias('Score'))
data.show(10)

data.columns

from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols = data.columns, outputCol = 'features')
final_data = assembler.transform(data)
final_data.show(5)

from pyspark.ml.clustering import KMeans

wcss = []
for i in range(2, 11):
    kmeans = KMeans(k = i, initMode = 'k-means||', maxIter = 100, initSteps = 10)
    # choose initMode = 'k-means||' to avoid random initialization trap
    # initMode = 'k-means||' uses k-means++ algorithm
    # maxIter = number of iteration the algorithm goes through to find optimal value of centroids
    # initSteps = number of times the K-means algorithm runs
    model = kmeans.fit(final_data)
    # Evaluating clustering by computing Within Set Sum of Squared Errors
    wssse = model.computeCost(final_data)
    wcss.append(wssse)   

# Visualizing the Elbow
import matplotlib.pyplot as plt

plt.plot(range(2, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Training k-means model with k=5
kmeans = KMeans(k = 5, initMode = 'k-means||', maxIter = 300, initSteps = 10)
model = kmeans.fit(final_data)

# Centers of theeach cluster
centers = model.clusterCenters()
print('Cluster Centers -------------')
for center in centers:
    print(center)

output = model.transform(final_data)
output.show(5)

# Convertind Spark dataframe to Pandas dataframe for visualization purpose
df = output.toPandas()
df.head()

df.prediction.unique()

import pandas as pd
cluster_center = pd.DataFrame(centers)

# Visualizing the clusters
plt.figure(figsize = (8, 6))
plt.scatter(df[df.prediction == 0]['Income'], df[df.prediction == 0]['Score'], s = 80, c = 'red', label = 'Cluster 1')
plt.scatter(df[df.prediction == 1]['Income'], df[df.prediction == 1]['Score'], s = 80, c = 'blue', label = 'Cluster 2')
plt.scatter(df[df.prediction == 2]['Income'], df[df.prediction == 2]['Score'], s = 80, c = 'green', label = 'Cluster 3')
plt.scatter(df[df.prediction == 3]['Income'], df[df.prediction == 3]['Score'], s = 80, c = 'cyan', label = 'Cluster 4')
plt.scatter(df[df.prediction == 4]['Income'], df[df.prediction == 4]['Score'], s = 80, c = 'magenta', label = 'Cluster 5')

# lets plot the centroids:
plt.scatter(cluster_center.iloc[:, 0], cluster_center.iloc[:, 1], s = 200, c = 'yellow', label = 'Centroids')

plt.title('Clusters of Clients')
plt.xlabel('Annual Income (K$)')
plt.ylabel('Spending Score(1-100)')
plt.legend(loc = 2, bbox_to_anchor=(1,1))
plt.show()



