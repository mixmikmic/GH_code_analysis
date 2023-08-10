import pandas as pd
import numpy as np
rand_st = 101
tst_sz=0.4

df_elu=pd.read_csv('elu_data.csv',usecols=['dropout', 'learning.rate', 'L1.regularization', 'training.steps',
       'accuracy', 'execution.time'])
df_elu.head()

def pred_quality(x):
    if x<0.5:
        return 0
    elif (x>=0.5 and x<0.8):
        return 1
    else:
        return 2

df_elu['Pred.Quality']=df_elu['accuracy'].apply(pred_quality)
df_elu.to_csv("elu_data_Quality.csv")

df_elu.head(5)

lst=list(df_elu.columns)

X=df_elu[lst[0:4]]
X.head()

y=df_elu['Pred.Quality']

from sklearn.linear_model import LogisticRegression
log_model=LogisticRegression(max_iter=10000,C=100,tol=0.0000001,solver='sag',multi_class='multinomial')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tst_sz,random_state=rand_st)

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
X_test_minmax = min_max_scaler.fit_transform(X_test)

log_model.fit(X_train_minmax,y_train)

y_pred=log_model.predict(X_test_minmax)

y_pred

from sklearn.metrics import confusion_matrix
conf_mat = pd.DataFrame(confusion_matrix(y_test, y_pred),
                        columns=['Predicted Low','Predicted Medium','Predicted High'], 
                        index=['True Low','True Medium','True High'])
conf_mat

from sklearn.metrics import classification_report
target_names = ['Low','Medium','High']
print(classification_report(y_test, y_pred, target_names=target_names))

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import logging
logging.set_verbosity(logging.ERROR)

def split_dataset(data,test_size):
    data_file = data
    data = pd.read_csv(data,usecols=['dropout', 'learning.rate', 'L1.regularization', 'training.steps','Pred.Quality'])
    from sklearn.model_selection import train_test_split
    data_train, data_test = train_test_split(data,test_size=test_size,random_state=rand_st)
    #data_train=pd.DataFrame(data_train)
    #data_test=pd.DataFrame(data_test)
    name = str(data_file).split('.')[0]
    name_train=name+'_train.csv'
    name_test=name+'_test.csv'
    data_train.to_csv(name_train,index=False,header=False)
    data_test.to_csv(name_test,index=False,header=False)
    return (data_train,data_test)

elu_train,elu_test=split_dataset('elu_data_Quality.csv',tst_sz)

training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename="elu_data_Quality_train.csv",
    target_dtype=np.int,
    features_dtype=np.float64,
    target_column=-1)

test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename='elu_data_Quality_test.csv',
    target_dtype=np.int,
    features_dtype=np.float64,
    target_column=-1)

# Specify that all features have real-value data
feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

# Define the training inputs
train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(training_set.data)},
      y=np.array(training_set.target),
      num_epochs=None,
      shuffle=True)

# Define the test inputs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(test_set.data)},
      y=np.array(test_set.target),
      num_epochs=1,
      shuffle=False)

classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,hidden_units=[20,20,20], dropout=0.01,
                                                      n_classes=3,
                                                    optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.001,
                                                    l1_regularization_strength=1.0),
                                                   activation_fn=tf.nn.elu)

classifier.train(input_fn=train_input_fn, steps=5000)
classifier.evaluate(input_fn=test_input_fn)["accuracy"]



