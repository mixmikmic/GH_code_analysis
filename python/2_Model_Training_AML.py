import os
from os import path, makedirs
import pandas as pd
import numpy as np

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from azureml.logging import get_azureml_logger
from pyspark.sql import SparkSession

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


# Fill in your Azure storage account information here
account_name = ''

# initialize logger
logger = get_azureml_logger()

spark = SparkSession.builder.getOrCreate()
data_filename = 'wasb://model@{}.blob.core.windows.net/trainingdata'.format(account_name)
df = spark.read.parquet(data_filename)

train, test = df.randomSplit([0.8, 0.2], seed=0)
train = train.sampleBy('label', fractions={0.0: 0.2, 1.0: 0.8}, seed=0)

#trained_model = RandomForestClassifier(featuresCol='features', labelCol='label').fit(train)

# Define the classifier   
clf = RandomForestClassifier(seed=0)
evaluator = BinaryClassificationEvaluator()
paramGrid = ParamGridBuilder().addGrid(clf.maxDepth, [5, 10]).addGrid(clf.maxBins, [32, 64]).build()

# Create 3-fold CrossValidator
cv = CrossValidator(estimator=clf, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)


# Run cross validations.  This can take up-to 5 minutes since there 2*2=4 parameter settings for each model, each of which trains with 3 traing set 
cvModel = cv.fit(train)

# Get the best model
trained_model = cvModel.bestModel

print("The evaluation metric is {}.".format(evaluator.getMetricName()))

print("Parameter MaxDepth of the best model is {}.".format(clf.getMaxDepth()))
print("Parameter MaxBins of the best model is {}.".format(clf.getMaxBins()))

logger.log("MaxDepth", (clf.getMaxDepth()))
logger.log("MaxBins", (clf.getMaxBins()))

# Store the model
model_filename = 'wasb://model@{}.blob.core.windows.net/model'.format(account_name)

trained_model.save(model_filename)

# Make predictions on test dataset. 
predictions = trained_model.transform(test)

# Evaluate the best trained model on the test dataset with default metric "areaUnderROC"
evaluator.evaluate(predictions)

# Create the confusion matrix for the multiclass prediction results
# This result assumes a decision boundary of p = 0.5

pred_pd = predictions.toPandas()
confuse = pd.crosstab(pred_pd['label'],pred_pd['prediction'])
confuse.columns = confuse.columns.map(str)
print(confuse)

# select (prediction, true label) and compute test error
# True positives - diagonal failure terms 
tp = confuse['1.0'][1]

# False positves - All failure terms - True positives
fp = np.sum(np.sum(confuse[['1.0']])) - tp

# True negatives 
tn = confuse['0.0'][0]

# False negatives total of non-failure column - TN
fn = np.sum(np.sum(confuse[['0.0']])) - tn


# Accuracy is diagonal/total 
acc_n = tn + tp
acc_d = np.sum(np.sum(confuse[['0.0','1.0']]))
acc = acc_n/acc_d

# Calculate precision and recall.
prec = tp/(tp+fp)
rec = tp/(tp+fn)

# Print the evaluation metrics to the notebook
print("Accuracy = %g" % acc)
print("Precision = %g" % prec)
print("Recall = %g" % rec )
print("F1 = %g" % (2.0 * prec * rec/(prec + rec)))
print("")

# logger writes information back into the AML Workbench run time page.
# Each title (i.e. "Model Accuracy") can be shown as a graph to track
# how the metric changes between runs.
logger.log("Model Accuracy", (acc))
logger.log("Model Precision", (prec))
logger.log("Model Recall", (rec))
logger.log("Model F1", (2.0 * prec * rec/(prec + rec)))

