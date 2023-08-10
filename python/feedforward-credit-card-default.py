

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

learning_rate = 0.01
training_epochs = 60
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

df = pd.read_csv(filename)

cc_training.show()

cc_training.select('balance','sex','education','marriage','age','default').describe().show()

cc_training = cc_training.select([col(c).cast("double") for c in cc_training.columns])


cc_training.show()

#count_classes = pd.value_counts(df['Class'], sort = True)
count_classes = pd.value_counts(cc_training.select('default').toPandas()['default'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Transaction class distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Status")
plt.ylabel("Frequency");

(trainingData, validData) = cc_training.select('balance','sex','education','marriage','age','default').randomSplit([.7,.3])

trainingData.groupBy('balance').count().sort('count', ascending=False ).show(40)

assembler =  VectorAssembler(inputCols=['balance','sex','education','marriage','age'], outputCol="assembled")
scaler = StandardScaler(inputCol="assembled", outputCol="features")
pipeline = Pipeline(stages = [assembler, scaler])
pipelineTraining = pipeline.fit(trainingData)
cc_data_training = pipelineTraining.transform(trainingData)
pipelineTest = pipeline.fit(validData)
cc_data_test = pipelineTest.transform(validData)

cc_data_training.select('features').rdd.map(lambda x: x[0]).map(lambda x: np.array(x)).take(2)

#convert ndarray data into RDD[Sample]

# balance,sex,education,marriage,age,default

def array2rdd(ds):
    #build Sample from ndarrays
    def build_sample(balance,sex,education,marriage,age,default):
        feature = np.array([balance,sex,education,marriage,age]).flatten()
        label = np.array(default)
        return Sample.from_ndarray(feature, label)
    rdd = ds.map(lambda (balance,sex,education,marriage,age,default): build_sample(balance,sex,education,marriage,age,default))
    return rdd


def DF2rdd(ds):
    #build Sample from ndarrays
    def build_sample(features, label):
        feature = np.array([balance,sex,education,marriage,age]).flatten()
        label = np.array(default)
        return Sample.from_ndarray(feature, label)
    features = ds.select('features').rdd.map(lambda x: x[0]).map(lambda x: np.array(x)).take(2)
    rdd = ds.map(lambda (balance,sex,education,marriage,age,default): build_sample(balance,sex,education,marriage,age,default))
    return rdd

cc_rdd_train = array2rdd(trainingData.rdd.map(list))
cc_rdd_train.cache()
cc_rdd_train.count()



cc_rdd_test = array2rdd(validData.rdd.map(list))
cc_rdd_test.cache()
cc_rdd_test.count()

validData.take(20)

cc_rdd_train.take(3)

# Create model

def multilayer_perceptron(n_hidden_1, n_hidden_2, n_input, n_classes):
    # Initialize a sequential container
    model = Sequential()
    # Hidden layer with ReLu activation
    model.add(Linear(n_input, n_hidden_1).set_name('mlp_fc1'))
    model.add(Dropout(0.2))
    model.add(ReLU())
    # Hidden layer with ReLu activation
    #model.add(Linear(n_hidden_1, n_hidden_2).set_name('mlp_fc2'))
    #model.add(ReLU())
    # output layer
    #model.add(Linear(n_hidden_2, n_classes).set_name('mlp_fc3'))
    model.add(Linear(n_hidden_1, n_classes).set_name('mlp_fc3'))
    model.add(LogSoftMax())
    return model

model = multilayer_perceptron(n_hidden_1, n_hidden_2, n_input, n_classes)

# Create an Optimizer
optimizer = Optimizer(
    model=model,
    training_rdd=cc_rdd_train,
    criterion=ClassNLLCriterion(),
    optim_method=Adagrad(learningrate=learning_rate, learningrate_decay=0.0002),
    end_trigger=MaxEpoch(training_epochs),
    batch_size=batch_size)

# Set the validation logic
optimizer.set_validation(
    batch_size=batch_size,
    val_rdd=cc_rdd_test,
    trigger=EveryEpoch(),
    val_method=[Top1Accuracy()]
)

app_name='cc-default-'+dt.datetime.now().strftime("%Y%m%d-%H%M%S")
train_summary = TrainSummary(log_dir='/tmp/bigdl_summaries',
                                     app_name=app_name)
train_summary.set_summary_trigger("Parameters", SeveralIteration(50))
val_summary = ValidationSummary(log_dir='/tmp/bigdl_summaries',
                                        app_name=app_name)
optimizer.set_train_summary(train_summary)
optimizer.set_val_summary(val_summary)
print("saving logs to ",app_name)

get_ipython().run_cell_magic('time', '', '# Boot training process\ntrained_model = optimizer.optimize()\nprint("Optimization Done.")')


loss = np.array(train_summary.read_scalar("Loss"))
top1 = np.array(val_summary.read_scalar("Top1Accuracy"))

plt.figure(figsize = (12,12))
plt.subplot(2,1,1)
plt.plot(loss[:,0],loss[:,1],label='loss')
plt.xlim(0,loss.shape[0]+10)
plt.grid(True)
plt.title("loss")
plt.subplot(2,1,2)
plt.plot(top1[:,0],top1[:,1],label='top1')
plt.xlim(0,loss.shape[0])
plt.title("top1 accuracy")
plt.grid(True)

predictions = trained_model.predict(cc_rdd_test).collect()

def map_predict_label(l):
    return np.array(l).argmax()
def map_groundtruth_label(l):
    return l.to_ndarray()[0] - 1

y_pred = np.array([ map_predict_label(s) for s in predictions])

y_true = np.array([map_groundtruth_label(s.label) for s in cc_rdd_test.collect()])

print(str(np.abs(predictions[0])))
print(str(y_true[0]))
map_predict_label(predictions[0])

acc = accuracy_score(y_true, y_pred)
print("The prediction accuracy is %.2f%%"%(acc*100))

cm = confusion_matrix(y_true, y_pred)
cm.shape
df_cm = pd.DataFrame(cm)
plt.figure(figsize = (10,8))
sn.heatmap(df_cm, annot=True,fmt='d');



