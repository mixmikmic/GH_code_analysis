sc

get_ipython().system('rm -rf metastore_db/*.lck')

from pyspark.sql import SQLContext
sqlc = SQLContext(sc)

from pyspark.ml.linalg import Vectors

df = sqlc.createDataFrame([(1.0, Vectors.dense(1.0, 2.0, 3.0)),
                           (1.0, Vectors.dense(2.0, 3.0, 4.0)),
                           (0.0, Vectors.dense(-1.0, 1.0, 2.0)),
                           (0.0, Vectors.dense(-2.0, 3.0, 5.0))]).toDF("label", "features")

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression()

lrModel = lr.fit(df)
lrModel.transform(df).show()

lrModel.save("lrModel.parquet")

from pyspark.ml.classification import LogisticRegressionModel

sameModel = LogisticRegressionModel.load("lrModel.parquet")
sameModel.transform(df).show()

get_ipython().system('cat lrModel.parquet/metadata/part-00000 ')

