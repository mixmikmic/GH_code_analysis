

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
from utils import *
from bigdl.models.ml_pipeline.dl_classifier import *


from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, udf
from pyspark.ml import  Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator




init_engine()

learning_rate = 0.1
training_epochs = 20
batch_size = 1024
display_step = 1

# Network Parameters
n_input = 5
n_classes = 2
n_hidden_1 = 3 # 1st layer number of features
n_hidden_2 = 2 # 1st layer number of features



filename =  "../data/cc-default/default-simple.csv"

LABELS = ["Good", "Default"] 

# Number of hidden layers

n_hidden_guess = np.sqrt(np.sqrt((n_classes + 2) * n_input) + 2 * np.sqrt(n_input /(n_classes+2.)))
print("Hidden layer 1 (Guess) : " + str(n_hidden_guess))

n_hidden_guess_2 = n_classes * np.sqrt(n_input / (n_classes + 2.))
print("Hidden layer 2 (Guess) : " + str(n_hidden_guess_2))

cc_training = spark.read.csv(filename, header=True, inferSchema="true", mode="DROPMALFORMED")
cc_training = cc_training.withColumn('label', cc_training.default.cast("double"))

cc_training.show()

cc_training.select('balance','sex','education','marriage','age','default').describe().show()

#count_classes = pd.value_counts(df['Class'], sort = True)
count_classes = pd.value_counts(cc_training.select('default').toPandas()['default'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Transaction class distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Status")
plt.ylabel("Frequency");

(trainingData, validData) = cc_training.select('balance','sex','education','marriage','age','label').randomSplit([.7,.3])

assembler =  VectorAssembler(inputCols=['balance','sex','education','marriage','age'], outputCol="assembled")
scaler = StandardScaler(inputCol="assembled", outputCol="features")
pipeline = Pipeline(stages = [assembler, scaler])
pipelineTraining = pipeline.fit(trainingData)
cc_data_training = pipelineTraining.transform(trainingData)
pipelineTest = pipeline.fit(validData)
cc_data_test = pipelineTest.transform(validData)

np.array(cc_data_training.select('features').collect())

bigDLModel = Sequential().add(Linear(n_input, n_hidden_1)).add(Linear(n_hidden_1, n_classes)).add(LogSoftMax())
classnll_criterion = ClassNLLCriterion()
dlClassifier = DLClassifier(model=bigDLModel, criterion=classnll_criterion, feature_size=[n_input])
dlClassifier.setLabelCol("default").setMaxEpoch(training_epochs).setBatchSize(batch_size)
model = dlClassifier.fit(cc_data_training)
print("\ninitial model training finished.")

from pyspark.sql import DataFrame, SQLContext
predictionDF = DataFrame(model.transform(cc_data_test), SQLContext(sc))
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

