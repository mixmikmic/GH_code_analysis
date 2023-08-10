spark.sparkContext.uiWebUrl

import matplotlib.pyplot as plt
import pandas as pd

get_ipython().magic('matplotlib inline')

df_training = (spark
               .read
               .options(header = False, inferSchema = True)
               .csv("data/MNIST/mnist_train.csv"))

df_training.count()

df_training.first()

print("No of columns: ", len(df_training.columns), df_training.columns)

feature_culumns = ["_c" + str(i+1) for i in range(784)]
print(feature_culumns)

from pyspark.ml.feature import VectorAssembler

vectorizer = VectorAssembler(inputCols=feature_culumns, outputCol="features")
training = (vectorizer
            .transform(df_training)
            .select("_c0", "features")
            .toDF("label", "features")
            .cache())
training.show()

a = training.first().features.toArray()
type(a)

plt.imshow(a.reshape(28, 28), cmap="Greys")

images = training.sample(False, 0.01, 1).take(25)
fig, _ = plt.subplots(5, 5, figsize = (10, 10))
for i, ax in enumerate(fig.axes):
    r = images[i]
    label = r.label
    features = r.features
    ax.imshow(features.toArray().reshape(28, 28), cmap = "Greys")
    ax.set_title("True: " + str(label))

plt.tight_layout()
    

counts = training.groupBy("label").count()

counts_df = counts.rdd.map(lambda r: {"label": r['label'], 
                                     "count": r['count']}).collect()
pd.DataFrame(counts_df).set_index("label").sort_index().plot.bar()

df_testing = (spark
              .read
              .options(header = False, inferSchema = True)
              .csv("data/MNIST/mnist_test.csv"))
testing = (vectorizer
           .transform(df_testing)
           .select("_c0", "features")
           .toDF("label", "features")
           .cache())

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(featuresCol="features", 
                        labelCol="label", 
                        regParam=0.1, 
                        elasticNetParam=0.1, 
                        maxIter=10000)

lr_model = lr.fit(training)

from pyspark.sql.functions import *

test_pred = lr_model.transform(testing).withColumn("matched", expr("label == prediction"))
test_pred.show()

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol="label", 
                                               predictionCol="prediction", 
                                               metricName="accuracy")

evaluator.evaluate(test_pred)

(test_pred
 .withColumn("matched", expr("cast(matched as int)"))
 .groupby("label")
 .agg(avg("matched"))
 .orderBy("label")
 .show())

from pyspark.ml.classification import MultilayerPerceptronClassifier

layers = [784, 100, 20, 10]
perceptron = MultilayerPerceptronClassifier(maxIter=1000, layers=layers, blockSize=128, seed=1234)
perceptron_model = perceptron.fit(training)

from time import time

start_time = time()
perceptron_model = perceptron.fit(training)
test_pred = perceptron_model.transform(testing)
print("Accuracy:", evaluator.evaluate(test_pred))
print("Time taken: %d" % (time() - start_time))



