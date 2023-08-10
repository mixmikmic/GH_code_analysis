get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc


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

from pyspark.sql.functions import col

LABELS = ["Setosa", "Virginica", "Versicolor"]


init_engine()

learning_rate = 0.1
training_epochs = 20
batch_size = 16
display_step = 1

# Network Parameters
n_input = 4   # Number of input dimensions
n_classes = 3  # Number of output classes (Setosa, Virginica, Versicolor)
n_hidden_1 = 3 # Hidden layer number of features

# Number of hidden layers

n_hidden_guess = np.sqrt(np.sqrt((n_classes + 2) * n_input) + 2 * np.sqrt(n_input /(n_classes+2.)))
print("Hidden layer 1 (Guess) : " + str(n_hidden_guess))

n_hidden_guess_2 = n_classes * np.sqrt(n_input / (n_classes + 2.))
print("Hidden layer 2 (Guess) : " + str(n_hidden_guess_2))

iris_training = spark.read.csv("../data/iris/iris_training.csv", header=True, inferSchema="true", mode="DROPMALFORMED")
iris_test = spark.read.csv("../data/iris/iris_test.csv", header=True, inferSchema="true", mode="DROPMALFORMED")

#Force everything to be of type double
iris_training = iris_training.select([col(c).cast("double") for c in iris_training.columns])
iris_test = iris_test.select([col(c).cast("double") for c in iris_test.columns])


iris_k_train = iris_training.rdd.map(list)
iris_k_test = iris_test.rdd.map(list)

count_classes = pd.value_counts(iris_training.select('label').toPandas()['label'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Training Class distribution")
plt.xticks(range(n_classes), LABELS)
plt.xlabel("Label")
plt.ylabel("Frequency");

count_classes = pd.value_counts(iris_test.select('label').toPandas()['label'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Validation Class distribution")
plt.xticks(range(n_classes), LABELS)
plt.xlabel("Label")
plt.ylabel("Frequency");

iris_k_train.take(29)

#convert ndarray data into RDD[Sample]
def array2rdd(ds):
    #build Sample from ndarrays
    def build_sample(c0,c1,c2,c3,prediction):
        feature = np.array([c0,c1, c2, c3]).flatten()
        label = np.array(prediction)
        return Sample.from_ndarray(feature, label)
    rdd = ds.map(lambda (c0,c1,c2,c3,prediction): build_sample(c0,c1,c2,c3,prediction))
    return rdd

iris_rdd_train = array2rdd(iris_k_train)
iris_rdd_train.cache()
print("Training Count: " + str(iris_rdd_train.count()))

iris_rdd_test = array2rdd(iris_k_test)
iris_rdd_test.cache()
print ("Test Count: " + str(iris_rdd_test.count()))

# Create model

def multilayer_perceptron(n_hidden_1, n_input, n_classes):
    # Initialize a sequential container
    model = Sequential()
    # Hidden layer with ReLu activation
    model.add(Linear(n_input, n_hidden_1).set_name('mlp_fc1'))
    model.add(ReLU())
    # output layer
    model.add(Linear(n_hidden_1, n_classes).set_name('mlp_fc4'))
    model.add(LogSoftMax())
    return model

model = multilayer_perceptron(n_hidden_1, n_input, n_classes)

# Create an Optimizer
optimizer = Optimizer(
    model=model,
    training_rdd=iris_rdd_train,
    criterion=ClassNLLCriterion(),
    optim_method=Adagrad(learningrate=learning_rate, learningrate_decay=0.0002),
    end_trigger=MaxEpoch(training_epochs),
    batch_size=batch_size)

# Set the validation logic
optimizer.set_validation(
    batch_size=batch_size,
    val_rdd=iris_rdd_test,
    trigger=EveryEpoch(),
    val_method=[Top1Accuracy()]
)

app_name='iris-'+dt.datetime.now().strftime("%Y%m%d-%H%M%S")
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
plt.xlim(0,loss.shape[0]+10)
plt.title("top1 accuracy")
plt.grid(True)

predictions = trained_model.predict(iris_rdd_test).collect()

def map_predict_label(l):
    return np.array(l).argmax()
def map_groundtruth_label(l):
    return l.to_ndarray()[0] - 1

y_pred = np.array([ map_predict_label(s) for s in predictions])

y_true = np.array([map_groundtruth_label(s.label) for s in iris_rdd_test.collect()])

acc = accuracy_score(y_true, y_pred)
print("The prediction accuracy is %.2f%%"%(acc*100))

cm = confusion_matrix(y_true, y_pred)
cm.shape
df_cm = pd.DataFrame(cm)
plt.figure(figsize = (10,8))
sn.heatmap(df_cm, annot=True,fmt='d');

def map_predict_label(outputs, threshold):
    p = map(lambda x: np.exp(x), outputs)
    if (np.max(p) < threshold):
        return len(outputs) + 1
    return np.argmax(outputs) + 1
    

true_positive_rate = list()
false_positive_rate = list()
 
for threshold in np.linspace(0.0, 1.0, num = 100):
    y_pred = np.array([map_predict_label(s, threshold) for s in predictions])
 



