import pandas as pd
import numpy as np
import tensorflow as tf

census = pd.read_csv('census_data.csv')

census.head()

census['income_bracket'].unique()

census['income_bracket'] = census['income_bracket'].apply(lambda x: 0 if x == ' <=50K' else 1)

census.head()

# Features
X = census.drop('income_bracket', axis=1)

# Labels
y = census['income_bracket']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

print(X_train)

X.columns

# Make Features
age = tf.feature_column.numeric_column('age')
edu_num = tf.feature_column.numeric_column('education_num')
capital_gain = tf.feature_column.numeric_column('capital_gain')
capital_loss = tf.feature_column.numeric_column('capital_loss')
hours_per_week = tf.feature_column.numeric_column('hours_per_week')

print(tf.__version__)

# To use DNNClassifier, conver the "Categorical Columns" to "Embedded Categorical Columns"

workclass = tf.feature_column.categorical_column_with_hash_bucket('workclass', hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket('education', hash_bucket_size=1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket('marital_status', hash_bucket_size=1000)
occupation = tf.feature_column.categorical_column_with_hash_bucket('occupation', hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket('relationship', hash_bucket_size=1000)
race = tf.feature_column.categorical_column_with_hash_bucket('race', hash_bucket_size=1000)
gender = tf.feature_column.categorical_column_with_hash_bucket('gender', hash_bucket_size=2)
native_country = tf.feature_column.categorical_column_with_hash_bucket('native_country', hash_bucket_size=1000)

feat_cols = [age, workclass, education, edu_num, marital_status, occupation, relationship, race, gender, capital_gain, capital_loss, hours_per_week, native_country]

input_func = tf.estimator.inputs.pandas_input_fn(X_train, y_train, batch_size=100, num_epochs=None, shuffle=True)

model = tf.estimator.LinearClassifier(feature_columns=feat_cols)

model.train(input_fn=input_func, steps=5000)

pred_input = tf.estimator.inputs.pandas_input_fn(X_test, batch_size=len(X_test), shuffle=False)

pred = model.predict(input_fn=pred_input)

predictions = list(pred)

predictions

final_preds = []

for ci in predictions:
    final_preds.append(ci['class_ids'][0])

print(len(final_preds))

final_preds

from sklearn.metrics import classification_report

print(classification_report(y_test, final_preds))

