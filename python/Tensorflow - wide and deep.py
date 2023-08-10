from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from six.moves import urllib

import pandas as pd
import tensorflow as tf

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]

def maybe_download(train_data, test_data):
  """Maybe downloads training data and returns train and test file names."""
  if train_data:
    train_file_name = train_data
  else:
    train_file = open('train.data', 'wb')
    urllib.request.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)  # pylint: disable=line-too-long
    train_file_name = train_file.name
    train_file.close()
    print("Training data is downloaded to %s" % train_file_name)

  if test_data:
    test_file_name = test_data
  else:
    test_file = open('test.data', 'wb')
    urllib.request.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)  # pylint: disable=line-too-long
    test_file_name = test_file.name
    test_file.close()
    print("Test data is downloaded to %s" % test_file_name)

  return train_file_name, test_file_name

train_file_name, test_file_name = maybe_download("train.data", "test.data")

def build_estimator(model_dir, model_type):
  """Build an estimator."""
  # Sparse base columns.
  combiner_type = 'sum'  
  gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender",
                                                     keys=["female", "male"], combiner=combiner_type)
  education = tf.contrib.layers.sparse_column_with_hash_bucket(
      "education", hash_bucket_size=1000, combiner=combiner_type)
  relationship = tf.contrib.layers.sparse_column_with_hash_bucket(
      "relationship", hash_bucket_size=100, combiner=combiner_type)
  workclass = tf.contrib.layers.sparse_column_with_hash_bucket(
      "workclass", hash_bucket_size=100, combiner=combiner_type)
  occupation = tf.contrib.layers.sparse_column_with_hash_bucket(
      "occupation", hash_bucket_size=1000, combiner=combiner_type)
  native_country = tf.contrib.layers.sparse_column_with_hash_bucket(
      "native_country", hash_bucket_size=1000, combiner=combiner_type)

  # Continuous base columns.
  age = tf.contrib.layers.real_valued_column("age")
  education_num = tf.contrib.layers.real_valued_column("education_num")
  capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
  capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
  hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")

  # Transformations.
  age_buckets = tf.contrib.layers.bucketized_column(age,
                                                    boundaries=[
                                                        18, 25, 30, 35, 40, 45,
                                                        50, 55, 60, 65
                                                    ])

  # Wide columns and deep columns.
  wide_columns = [gender, native_country, education, occupation, workclass,
                  relationship, age_buckets,
                  tf.contrib.layers.crossed_column([education, occupation],
                                                   hash_bucket_size=int(1e4)),
                  tf.contrib.layers.crossed_column(
                      [age_buckets, education, occupation],
                      hash_bucket_size=int(1e6)),
                  tf.contrib.layers.crossed_column([native_country, occupation],
                                                   hash_bucket_size=int(1e4))]
  dims = 8 
  deep_columns = [
      tf.contrib.layers.embedding_column(workclass, dimension=dims, combiner=combiner_type),
      tf.contrib.layers.embedding_column(education, dimension=dims, combiner=combiner_type),
      tf.contrib.layers.embedding_column(gender, dimension=dims, combiner=combiner_type),
      tf.contrib.layers.embedding_column(relationship, dimension=dims, combiner=combiner_type),
      tf.contrib.layers.embedding_column(native_country,
                                         dimension=dims, combiner=combiner_type),
      tf.contrib.layers.embedding_column(occupation, dimension=dims, combiner=combiner_type),
      age,
      education_num,
      capital_gain,
      capital_loss,
      hours_per_week,
  ]

  if model_type == "wide":
    m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                          feature_columns=wide_columns)
  elif model_type == "deep":
    m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                       feature_columns=deep_columns,
                                       hidden_units=[100, 50])
  else:
    m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50])
  return m

def input_fn(df):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {
      k: tf.SparseTensor(
          indices=[[i, 0] for i in range(df[k].size)],
          values=df[k].values,
          dense_shape=[df[k].size, 1])
      for k in CATEGORICAL_COLUMNS}

  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label

model_type = 'wide_n_deep' #{'wide', 'deep', 'wide_n_deep'}
train_steps = 400

df_train = pd.read_csv(
  tf.gfile.Open(train_file_name), names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(
  tf.gfile.Open(test_file_name), names=COLUMNS, skipinitialspace=True, skiprows=1)

# remove NaN elements
df_train = df_train.dropna(how='any', axis=0)
df_test = df_test.dropna(how='any', axis=0)

df_train[LABEL_COLUMN] = (
  df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
df_test[LABEL_COLUMN] = (
  df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

tf.logging.set_verbosity(tf.logging.ERROR)
m = build_estimator('output_models', model_type)
m.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)
results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
for key in sorted(results):
  print("%s: %s" % (key, results[key]))

