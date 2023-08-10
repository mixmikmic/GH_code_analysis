import pyspark
from pyspark.context import SparkContext
from pyspark.sql import SparkSession, SQLContext

spark = SparkSession.builder.master("local[1]").getOrCreate()
sc = spark.sparkContext

from pyspark.sql.functions import array, column, rand, udf
from pyspark.ml.linalg import Vectors, VectorUDT
as_vector = udf(lambda l: Vectors.dense(l), VectorUDT())

randomDF = spark.range(0, 2048).select((rand() * 2 - 1).alias("x"), (rand() * 2 - 1).alias("y")).select(column("x"), column("y"), as_vector(array(column("x"), column("y"))).alias("features"))

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'svg'")

import seaborn as sns
import numpy as np
sns.set(color_codes=True)

_ = sns.lmplot("x", "y", randomDF.toPandas(), fit_reg=False, scatter=True)

from pyspark.ml.clustering import KMeans

K = 7
SEED = 0xdea110c8

kmeans = KMeans().setK(K).setSeed(SEED).setFeaturesCol("features")
model = kmeans.fit(randomDF)

withPredictions = model.transform(randomDF).select("x", "y", "prediction")

_ = sns.lmplot("x", "y", withPredictions.toPandas(), fit_reg=False, scatter=True, hue="prediction")

from pyspark.sql.functions import count
withPredictions.groupBy("prediction").agg(count("prediction")).show()

def cluster_and_plot(df, k, seed=0xdea110c8):
    kmeans = KMeans().setK(k).setSeed(seed).setFeaturesCol("features")
    withPredictions = kmeans.fit(df).transform(df).select("x", "y", "prediction")
    return sns.lmplot("x", "y", withPredictions.toPandas(), fit_reg=False, scatter=True, hue="prediction")

_ = cluster_and_plot(randomDF, 11)

from pyspark.sql.types import DoubleType
from random import uniform

synthetic_label = udf(lambda v: (abs(v[0]) * v[0]) + (v[1] * 2) + uniform(-0.5, 0.5) > 0 and 1.0 or 0.0, DoubleType())

labeledDF = randomDF.withColumn("label", synthetic_label(randomDF["features"]))
_ = sns.lmplot("x", "y", labeledDF.select("x", "y", "label").toPandas(), hue="label", fit_reg=False)

from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression()
training, test = labeledDF.randomSplit([.7,.3])

lr_model = lr.fit(training)

lr_predictions = lr_model.transform(labeledDF)

lr_predictions.printSchema()

_ = sns.lmplot("x", "y", lr_predictions.select("x", "y", "prediction").toPandas(), hue="prediction", fit_reg=False)

_ = sns.lmplot("x", "y", lr_predictions.filter(lr_predictions["prediction"] != lr_predictions["label"]).select("x", "y", "label").toPandas(), hue="label", fit_reg=False).set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))

from pyspark.sql.functions import column, sum, when
lrp = lr_predictions
lr_predictions.select("label", when(lrp["prediction"] == 0.0, 1).otherwise(0).alias("p0"), when(lrp["prediction"] == 1.0, 1).otherwise(0).alias("p1")).groupBy("label").agg(sum(column("p0")).alias("predicted 0"), sum(column("p1")).alias("predicted 1")).show()

summary = lr_model.summary
roc = summary.roc
roc = roc.select(roc["FPR"].alias("False Positive Rate"), roc["TPR"].alias("True Positive Rate")).toPandas()

_ = sns.lmplot("False Positive Rate", "True Positive Rate", roc, fit_reg=False, scatter=True, scatter_kws={'marker':'1', 's':6}).set(xlim=(0,1), ylim=(0,1))

mpg = spark.read.json("data/auto-mpg.json")
mpg.printSchema()

mpg_with_features = mpg.dropna().select("mpg", "acceleration", "cylinders", "displacement", "horsepower", "weight", as_vector(array(mpg["acceleration"], mpg["cylinders"], mpg["displacement"], mpg["horsepower"], mpg["weight"])).alias("features"))

from pyspark.ml.regression import LinearRegression

lr = LinearRegression()
lr.setLabelCol("mpg")
lr.setFeaturesCol("features")
lr.setStandardization(True)
model = lr.fit(mpg_with_features)

predictions = model.transform(mpg_with_features)
to_plot = predictions.select("mpg", "acceleration", "cylinders", "displacement", "horsepower", "weight", "prediction").show()



