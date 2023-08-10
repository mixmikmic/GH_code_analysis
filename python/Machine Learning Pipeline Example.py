sc

sqlContext

df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").load("resources/adult.data")

df.cache()

df.head()

df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("resources/adult.data")

df.head()

df.cache()

from pyspark.mllib.linalg import Vectors
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.param import Param, Params
from pyspark.ml.feature import Bucketizer, VectorAssembler, StringIndexer
from pyspark.ml import Pipeline

assembler = VectorAssembler(inputCols=["age", "education-num"], outputCol="feautres")

indexer = StringIndexer(inputCol="category").setOutputCol("category-index")

pipeline = Pipeline().setStages([assembler, indexer])

model=pipeline.fit(df)

prepared = model.transform(df)

prepared.head()

dt = DecisionTreeClassifier(labelCol = "category-index", featuresCol="feautres")

dt_model = dt.fit(prepared)

dt_model

pipeline_and_model = Pipeline().setStages([assembler, indexer, dt])
pipeline_model = pipeline_and_model.fit(df)

dt_model.transform(prepared).select("prediction", "category-index").take(20)

pipeline_model.transform(df).select("prediction", "category-index").take(20)

labels = list(pipeline_model.stages[1].labels)

from pyspark.ml.feature import IndexToString
inverter = IndexToString(inputCol="prediction", outputCol="prediction-label", labels=labels)

inverter.transform(pipeline_model.transform(df)).select("prediction-label", "category").take(20)

pipeline_model.stages[2]

from pyspark.sql.functions import *
df.groupBy("age").agg(min("hours-per-week"), avg("hours-per-week"), max("capital-gain"))

from pyspark.sql.window import Window
windowSpec = Window.partitionBy("age").orderBy("capital-gain").rowsBetween(-100, 100)

df.select(df["age"], df['capital-gain'], avg("capital-gain").over(windowSpec)).orderBy(desc("capital-gain")).show()



