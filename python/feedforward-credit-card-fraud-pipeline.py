get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sn
import pandas as pd
import random as rd
import datetime as dt


from bigdl.dataset.transformer import *
from bigdl.dataset.base import *
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from bigdl.models.ml_pipeline.dl_classifier import *
from utils import *

from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
from pyspark.ml import  Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


init_engine()

learning_rate = 0.1
training_epochs = 100
batch_size = 10000
display_step = 1

# Network Parameters
n_input = 29
n_classes = 2
n_hidden_1 = 10 # 1st layer number of features


LABELS = ["Normal", "Fraud"]

cols = ["V" + str(x) for x in list(range(1,29))] + ["Amount"]
cols

cc_training = spark.read.csv("../data/creditcardfraud/creditcard.csv", header=True, inferSchema="true", mode="DROPMALFORMED")

cc_training.select('Time', 'V1', 'V2', 'Amount', 'Class').describe().show()

cc_training = cc_training.select([col(c).cast("double") for c in cc_training.columns])
cc_training = cc_training.withColumn("label", cc_training["Class"])

#count_classes = pd.value_counts(df['Class'], sort = True)
count_classes = pd.value_counts(cc_training.select('Class').toPandas()['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Transaction class distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency");

# get the time to split the data.
splitTime = cc_training.stat.approxQuantile("Time", [0.7], 0.001)[0]

trainingData =cc_training.filter("Time < " + str(splitTime))
validData = cc_training.filter("Time >= " + str(splitTime))

assembler =  VectorAssembler(inputCols=cols, outputCol="assembled")
scaler = StandardScaler(inputCol="assembled", outputCol="features")
pipeline = Pipeline(stages = [assembler, scaler])
pipelineTraining = pipeline.fit(trainingData)
data_training = pipelineTraining.transform(trainingData)
pipelineTest = pipeline.fit(validData)
data_test = pipelineTest.transform(validData)




bigDLModel = Sequential().add(Linear(n_input, n_hidden_1)).add(Linear(n_hidden_1, n_classes)).add(LogSoftMax())
classnll_criterion = ClassNLLCriterion()
dlClassifier = DLClassifier(model=bigDLModel, criterion=classnll_criterion, feature_size=[n_input])
dlClassifier.setLabelCol("label").setMaxEpoch(training_epochs).setBatchSize(batch_size)
model = dlClassifier.fit(data_training)
print("\ninitial model training finished.")

from pyspark.sql import DataFrame, SQLContext
predictionDF = DataFrame(model.transform(data_test), SQLContext(sc))
predictionDF

predictionDF.cache() 
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
auPRC = evaluator.evaluate(predictionDF)
print("\nArea under precision-recall curve: = " + str(auPRC))
    
recall = MulticlassClassificationEvaluator(metricName="weightedRecall").evaluate(predictionDF)
print("\nrecall = " + str(recall))

precision = MulticlassClassificationEvaluator(metricName="weightedPrecision").evaluate(predictionDF)
print("\nPrecision = " + str(precision))  
predictionDF.unpersist()

y_pred = np.array(predictionDF.select('prediction').collect())
y_true = np.array(predictionDF.select('label').collect())

acc = accuracy_score(y_true, y_pred)
print("The prediction accuracy is %.2f%%"%(acc*100))

cm = confusion_matrix(y_true, y_pred)
cm.shape
df_cm = pd.DataFrame(cm)
plt.figure(figsize = (10,8))
sn.heatmap(df_cm, annot=True,fmt='d');



