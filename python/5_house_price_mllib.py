sqlContext = SQLContext(sc)

text_RDD = sc.textFile("/data/houses.txt")

def mapper_parse_lines(line):
    """Parse line into (neighborhoood, price) pair"""
    words = line.split()
    return (words[1], float(words[2]), int(words[0]))

house_prices_RDD = text_RDD.map(mapper_parse_lines)

house_prices_RDD.collect()

house_prices_df = sqlContext.createDataFrame(house_prices_RDD, ["neighborhood", "price", "bedrooms"])

house_prices_df.show()

house_prices_df.printSchema()

from pyspark.sql import Row
from pyspark.mllib.linalg import Vectors

def create_features(row):
    return Row(neighborhood=row.neighborhood,
               features=Vectors.dense([row.bedrooms, row.price]))
    
house_prices_features = sqlContext.createDataFrame(
    house_prices_df.map(create_features))

house_prices_features.show()

from pyspark.ml.clustering import KMeans

kmeans = KMeans()

print(kmeans.explainParams())

model = kmeans.fit(house_prices_features)
centers = model.clusterCenters()

centers

transformed = model.transform(house_prices_features)

transformed.collect()

new_houses = sqlContext.createDataFrame([
    (Vectors.dense([3.0, 450000]),),
    (Vectors.dense([2.0, 500000]),),        
        ],
    ["features"]
)

new_houses.show()

model.transform(new_houses).collect()

from pyspark.ml.classification import LogisticRegression

from pyspark.ml.feature import StringIndexer

stringIndexer = StringIndexer(inputCol="neighborhood",
                              outputCol="label")

house_prices_features_labels = stringIndexer.fit(house_prices_features).transform(house_prices_features)

house_prices_features_labels.collect()

# Create a LogisticRegression instance. This instance is an Estimator.
lr = LogisticRegression(maxIter=10, regParam=0.01)
# Print out the parameters, documentation, and any default values.
print "LogisticRegression parameters:\n" + lr.explainParams() + "\n"

# Learn a LogisticRegression model. This uses the parameters stored in lr.
model1 = lr.fit(house_prices_features_labels)

model1.transform(house_prices_features).collect()

model1.transform(new_houses).collect()



