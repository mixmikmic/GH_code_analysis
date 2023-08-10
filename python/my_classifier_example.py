import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

diabetes = pd.read_csv('pima-indians-diabetes.csv')
diabetes.head()

diabetes.columns

cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
       'Insulin', 'BMI', 'Pedigree']

#normalize columns, using pandas trick
diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x-x.min())/(x.max()-x.min()))

diabetes.head()

#continuous features
num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
# can be treated as a continuous column
age = tf.feature_column.numeric_column('Age')

#we know that there'll be only 4 groups
assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group',['A','B','C','D'])

#alternative
#assigned_group = tf.feature_column.categorical_column_with_hash_bucket('Group',hash_bucket_size=10)

diabetes['Age'].hist(bins=20)

age_bucketized = tf.feature_column.bucketized_column(age,boundaries=[20,30,40,50,60,70,80])

#features columns
feat_cols =[num_preg,plasma_gluc,dias_press,tricep,insulin,bmi,diabetes_pedigree,assigned_group, age_bucketized]

#train-test split
x_data = diabetes.drop('Class',axis=1)
x_data.head()

labels = diabetes['Class']
#labels

#train-test split
x_train, x_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.3, random_state=101)

#We're going to use pandas input function instead of numpy because our data is already in a pandas dataframe
input_func = tf.estimator.inputs.pandas_input_fn(x = x_train, y = y_train, batch_size=10, num_epochs=1000, shuffle=True)

#create the Linear Classifier model built in Tensorflow
model = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=2)

#training
model.train(input_fn=input_func, steps=1000)

eval_input_func = tf.estimator.inputs.pandas_input_fn(x=x_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)

results = model.evaluate(input_fn=eval_input_func)

pred_input_func = tf.estimator.inputs.pandas_input_fn(x=x_test, batch_size=10, num_epochs=1, shuffle=False)

predictions = model.predict(pred_input_func)
my_pred = list(predictions)
my_pred

dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10],feature_columns=feat_cols,n_classes=2)

embedded_group_col = tf.feature_column.embedding_column(assigned_group, dimension=4)

feat_cols = [num_preg,plasma_gluc,dias_press,tricep,insulin,bmi,diabetes_pedigree, embedded_group_col, age_bucketized]

input_func = tf.estimator.inputs.pandas_input_fn(x=x_train,y=y_train,batch_size=10,num_epochs=1,shuffle=True)

dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10],feature_columns=feat_cols,n_classes=2)

dnn_model.train(input_fn=input_func,steps=10000)

eval_input_func = tf.estimator.inputs.pandas_input_fn(x=x_test,y=y_test,batch_size=10,num_epochs=1,shuffle=False)

dnn_model.evaluate(eval_input_func)

results



