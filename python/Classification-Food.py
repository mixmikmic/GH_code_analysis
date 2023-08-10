# Import Spark SQL and Spark ML libraries
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator

# Load the source data
csv = sqlContext.sql("Select * from food2")

# Select features and label
# Logistic Regression
data = csv.select("review_count","Take-out", "GoodFor_lunch", "GoodFor_dinner", "GoodFor_breakfast","Noise_Level", "Takes_Reservations","Delivery","Parking_lot", "WheelchairAccessible","Alcohol", "WaiterService","Wi-Fi","stars")

data.show(5)

def indexStringColumns(df, cols):
    #variable newdf will be updated several times
    newdata = df
    for c in cols:
        si = StringIndexer(inputCol=c, outputCol=c+"-x")
        sm = si.fit(newdata)
        newdata = sm.transform(newdata).drop(c)
        newdata = newdata.withColumnRenamed(c+"-x", c)
    return newdata

dfnumeric = indexStringColumns(data, ["Take-out","GoodFor_lunch", "GoodFor_dinner", "GoodFor_breakfast","Noise_Level", "Takes_Reservations","Delivery","Parking_lot", "WheelchairAccessible","Alcohol", "WaiterService","Wi-Fi"])


dfnumeric.show(25)

def oneHotEncodeColumns(df, cols):
    from pyspark.ml.feature import OneHotEncoder
    newdf = df
    for c in cols:
        onehotenc = OneHotEncoder(inputCol=c, outputCol=c+"-onehot", dropLast=False)
        newdf = onehotenc.transform(newdf).drop(c)
        newdf = newdf.withColumnRenamed(c+"-onehot", c)
    return newdf

dfhot = oneHotEncodeColumns(dfnumeric, ["Take-out","GoodFor_lunch", "GoodFor_dinner", "GoodFor_breakfast","Noise_Level", "Takes_Reservations","Delivery","Parking_lot", "WheelchairAccessible","Alcohol", "WaiterService","Wi-Fi"])

dfhot.show(25)

va = VectorAssembler(outputCol="features", inputCols=list(set(dfhot.columns)-set(['stars'])))
lpoints = va.transform(dfhot).select("features", "stars").withColumnRenamed("stars","label")

# Split the data
splits = lpoints.randomSplit([0.8, 0.2])
adulttrain = splits[0].cache()
adultvalid = splits[1].cache()

lr = LogisticRegression(regParam=0.01, maxIter=1000, fitIntercept=True)
lrmodel = lr.fit(adulttrain)
lrmodel = lr.setParams(regParam=0.01, maxIter=500, fitIntercept=True).fit(adulttrain)
lrmodel.intercept

validpredicts = lrmodel.transform(adultvalid)

validpredicts.show(5)

from pyspark.ml.evaluation import BinaryClassificationEvaluator
bceval = BinaryClassificationEvaluator()
bceval.evaluate(validpredicts)
bceval.getMetricName()

bceval.setMetricName("areaUnderPR")
bceval.evaluate(validpredicts)

display(validpredicts)

from pyspark.ml.tuning import CrossValidator
cv = CrossValidator().setEstimator(lr).setEvaluator(bceval).setNumFolds(2)
paramGrid = ParamGridBuilder().addGrid(lr.maxIter, [1000]).addGrid(lr.regParam, [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]).build()
cv.setEstimatorParamMaps(paramGrid)
cvmodel = cv.fit(adulttrain)

BinaryClassificationEvaluator().evaluate(cvmodel.bestModel.transform(adultvalid))

# LogisticRegression with attribute 'threshold' in ParamGridBuilder and BinaryClassificationEvaluator
paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.3, 0.1, 0.01]).addGrid(lr.maxIter, [10, 5]).addGrid(lr.threshold, [0.35, 0.30]).build()

tvs = TrainValidationSplit(estimator=lr, evaluator=RegressionEvaluator(), estimatorParamMaps=paramGrid, trainRatio=0.8)
model = tvs.fit(adulttrain)

prediction = model.transform(adultvalid)
# LogisticRegression
predicted = prediction.select("features", "prediction", "probability", "label")

predicted.show(100)

# Only for Classification Logistic Regression 

tp = float(predicted.filter("prediction == 1.0 AND label == 1").count())
fp = float(predicted.filter("prediction == 1.0 AND label == 0").count())
tn = float(predicted.filter("prediction == 0.0 AND label == 0").count())
fn = float(predicted.filter("prediction == 0.0 AND label == 1").count())
metrics = spark.createDataFrame([
 ("TP", tp),
 ("FP", fp),
 ("TN", tn),
 ("FN", fn),
 ("Precision", tp / (tp + fp)),
 ("Recall", tp / (tp + fn))],["metric", "value"])
metrics.show()

display(metrics)

evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC")
aur = evaluator.evaluate(validpredicts)
print "AUR = ", aur



