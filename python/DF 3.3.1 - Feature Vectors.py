sc

get_ipython().system('rm -rf metastore_db/*.lck')
from pyspark.sql import SQLContext
sqlc = SQLContext(sc)

from collections import namedtuple

Customer = namedtuple('Customer', ['churn','sessions','revenue','recency'])

customers = sc.parallelize([Customer(1, 20, 61.24, 103),
                            Customer(1, 8, 80.64, 23),
                            Customer(0, 4, 100.94, 42),
                            Customer(0, 8, 99.48, 26),
                            Customer(1, 17, 120.56, 47)]).toDF()

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler().setInputCols(["sessions", "revenue", "recency"]).setOutputCol("features")
dfWithFeatures = assembler.transform(customers)

dfWithFeatures.show()

from pyspark.ml.feature import VectorSlicer
slicer = VectorSlicer().setInputCol("features").setOutputCol("some_features")

slicer.setIndices([0, 1]).transform(dfWithFeatures).show()



