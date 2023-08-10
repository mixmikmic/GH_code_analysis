import pyspark

# Create a configuration object.
conf = (
    pyspark
      .SparkConf()
      .setMaster('local[*]')
      .setAppName('Simple Decision Tree Notebook')
)

# Create a Spark context for local work
try:
    sc
except:
    sc = pyspark.SparkContext(conf = conf)
    
print('Running with Spark version: ',sc.version)

import os
import urllib.request

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'
localfile = 'data/covtype.data.gz'

# Ensure we fetch the data if we need to.
if(not os.path.isfile(localfile)):
    print("Downloading data into: ",localfile)
    localfile, headers = urllib.request.urlretrieve(url, localfile)
else:
    print("Data file already present at: ",localfile)

rawData = sc.textFile(localfile)

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

# Extract the dataset features and target.
def ingest(line):
    # Simple numeric features (some are one-hot encoded).
    # Last field is the label (training target).
    fields = [float(f) for f in line.split(',')]
    features = Vectors.dense(fields[0:len(fields)-1])
    
    # Subtract 1 from the label to satisfy the '0' based
    # DecisionTree model.
    label    = fields[-1] - 1
    return LabeledPoint(label,features)

pointdata = rawData.map(ingest)

pointdata.take(5)

trainData, cvData, testData = pointdata.randomSplit([0.8,0.1,0.1])
trainData.cache()
cvData.cache()
testData.cache()

print(trainData.count(),cvData.count(),testData.count())

from pyspark.mllib.tree import DecisionTree, DecisionTreeModel

model = DecisionTree.trainClassifier( trainData, 7, {}, "gini", 4, 100)

# Score the validation data using the model.
cvPredictions = model.predict( cvData.map(lambda x: x.features))

# Label the results.
cvLabelsAndPredictions = cvData.map(lambda x: x.label).zip(cvPredictions)

# Create a metrics object to evaluate the results.
from pyspark.mllib.evaluation import MulticlassMetrics
cvMetrics = MulticlassMetrics(cvLabelsAndPredictions)

print('                 precision: ',cvMetrics.precision(),'\n',
      '                   recall: ',cvMetrics.recall(),'\n',
      '                 fMeasure: ',cvMetrics.fMeasure(),'\n',
      '        weightedPrecision: ',cvMetrics.weightedPrecision,'\n',
      '           weightedRecall: ',cvMetrics.weightedRecall,'\n',
      'weightedFalsePositiveRate: ',cvMetrics.weightedFalsePositiveRate,'\n',
      ' weightedTruePositiveRate: ',cvMetrics.weightedTruePositiveRate,'\n',
      '\nConfusion Matrix:'
     )

import numpy as np
print(cvMetrics.confusionMatrix().toArray().astype(int))

# Score and label the test data using the model.
testPredictions = model.predict( testData.map(lambda x: x.features))
testLabelsAndPredictions = testData.map(lambda x: x.label).zip(testPredictions)

# Create a metrics object for the test results.
testMetrics = MulticlassMetrics(testLabelsAndPredictions)

print('                 precision: ',testMetrics.precision(),'\n',
      '                   recall: ',testMetrics.recall(),'\n',
      '                 fMeasure: ',testMetrics.fMeasure(),'\n',
      '        weightedPrecision: ',testMetrics.weightedPrecision,'\n',
      '           weightedRecall: ',testMetrics.weightedRecall,'\n',
      'weightedFalsePositiveRate: ',testMetrics.weightedFalsePositiveRate,'\n',
      ' weightedTruePositiveRate: ',testMetrics.weightedTruePositiveRate,'\n',
      '\nConfusion Matrix:'
     )

print(testMetrics.confusionMatrix().toArray().astype(int))

# Location where the model will be stored.
modelLocation = 'tree-model'

# Ensure that there is no model currently in the location.
# Choose to store multiple models by using multiple locations.
import shutil
shutil.rmtree(modelLocation,ignore_errors=True)

# Actually save the model.
model.save(sc,modelLocation)

sameModel = DecisionTreeModel.load(sc,modelLocation)

otherData = pointdata.sample(False,0.2,2112)
otherData.cache()
print('Samples to predict with previously stored model: ',otherData.count())

otherPredictions = model.predict( otherData.map(lambda x: x.features))
otherLabelsAndPredictions = otherData.map(lambda x: x.label).zip(otherPredictions)
otherMetrics = MulticlassMetrics(otherLabelsAndPredictions)

print('                 precision: ',otherMetrics.precision(),'\n',
      '                   recall: ',otherMetrics.recall(),'\n',
      '                 fMeasure: ',otherMetrics.fMeasure(),'\n',
      '        weightedPrecision: ',otherMetrics.weightedPrecision,'\n',
      '           weightedRecall: ',otherMetrics.weightedRecall,'\n',
      'weightedFalsePositiveRate: ',otherMetrics.weightedFalsePositiveRate,'\n',
      ' weightedTruePositiveRate: ',otherMetrics.weightedTruePositiveRate,'\n',
      '\nConfusion Matrix:'
     )

print(otherMetrics.confusionMatrix().toArray().astype(int))

