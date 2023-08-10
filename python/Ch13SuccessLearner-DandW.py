#Copyright April 1, 2018 Warren E. Agin
#This code is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
#You may obtain a copy of the license at https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

#This code is derived from sample code available at https://www.tensorflow.org/tutorials/wide,
#licensed under the Apache License, Version 2.0. The license notice for the original
#code is provided below.

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#THE FILES HAVE BEEN CHANGED FROM THE VERSION RELEASED BY THE TENSORFLOW AUTHORS


DATA_URL = ''
TRAINING_FILE = 'trainingFile.csv'
TRAINING_URL = '%s/%s' % (DATA_URL, TRAINING_FILE)
EVAL_FILE = 'testFile.csv'
EVAL_URL = '%s/%s' % (DATA_URL, EVAL_FILE)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys
import csv

import tensorflow as tf

#create feature column names and default types using featureNames.csv file

with open('featureNames.csv', 'r', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        _CSV_COLUMNS = row
print(_CSV_COLUMNS)        
#dtypes for each field, needs to be defined manually        
_CSV_COLUMN_DEFAULTS=[[0],[0],[0],[0],[0],[0.0],[0],[0],[0],[0],[0],[0.0],[0.0],[0.0],[0],[0.0],[0.0],[0.0],[0],[0.0],[0.0],[0.0],[0],[0],[0.0],[0.0],[0.0],[0],[0.0],[0.0],[0.0],[0.0],[0.0]]   

#set up variables for use with the model

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir', type=str, default='/Users/trustee/Documents/Transport/Temp for 13 project/model_data',
    help='Base directory for the model.')

parser.add_argument(
    '--model_type', type=str, default='deep',
    help="Valid model types: {'wide', 'deep', 'wide_deep'}.")

parser.add_argument(
    '--train_epochs', type=int, default=50, help='Number of training epochs.') 

parser.add_argument(
    '--epochs_per_eval', type=int, default=2,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--batch_size', type=int, default=20, help='Number of examples per batch.')

parser.add_argument(
    '--train_data', type=str, default='trainingFile.csv',  #revise to match your location and directory structure
    help='Path to the training data.')

parser.add_argument(
    '--test_data', type=str, default='testFile.csv',  #revise to match your location and directory structure
    help='Path to the test data.')

_l1_strength=0.0
#Strength of the l1 regularization for the wide model - 0 to 1. 0 is off

_l2_strength=0.0
#Strength of the l2 regularization for the wide model - 0 to 1. 0 is off

#need to manually fill in dataset sizes. Information is in LEARNER_README.txt

_NUM_EXAMPLES = {
    'train': 400000,
    'validation': 30000,
}


def build_model_columns():
    """Builds a set of wide and deep feature columns."""
    # Continuous columns
    NTRDBT=tf.feature_column.categorical_column_with_identity('NTRDBT', num_buckets=2)
    JOINT=tf.feature_column.categorical_column_with_identity('JOINT', num_buckets=2)
    ORGD1FPRSE=tf.feature_column.categorical_column_with_identity('ORGD1FPRSE', num_buckets=2)
    PRFILE=tf.feature_column.categorical_column_with_identity('PRFILE', num_buckets=2)
    DISTSUCCESS=tf.feature_column.numeric_column('DISTSUCCESS')
    FEEP=tf.feature_column.categorical_column_with_identity('FEEP', num_buckets=2)
    FEEI=tf.feature_column.categorical_column_with_identity('FEEI', num_buckets=2)
    FEEW=tf.feature_column.categorical_column_with_identity('FEEW', num_buckets=2)
    REALPROPNULL=tf.feature_column.categorical_column_with_identity('REALPROPNULL', num_buckets=2)
    REALPROPNONE=tf.feature_column.categorical_column_with_identity('REALPROPNONE', num_buckets=2)
    REALPROPVALUE=tf.feature_column.numeric_column('REALPROPVALUE')
    REALPROPVALUESQR=tf.feature_column.numeric_column('REALPROPVALUESQR')
    REALPROPVALUELOG=tf.feature_column.numeric_column('REALPROPVALUELOG')
    PERSPROPNULL=tf.feature_column.categorical_column_with_identity('PERSPROPNULL', num_buckets=2)
    PERSPROPVALUE=tf.feature_column.numeric_column('PERSPROPVALUE')
    PERSPROPVALUESQR=tf.feature_column.numeric_column('PERSPROPVALUESQR')
    PERSPROPVALUELOG=tf.feature_column.numeric_column('PERSPROPVALUELOG')
    UNSECNPRNULL=tf.feature_column.categorical_column_with_identity('UNSECNPRNULL', num_buckets=2)
    UNSECNPRVALUE=tf.feature_column.numeric_column('UNSECNPRVALUE')
    UNSECNPRVALUESQR=tf.feature_column.numeric_column('UNSECNPRVALUESQR')
    UNSECNPRVALUELOG=tf.feature_column.numeric_column('UNSECNPRVALUELOG')
    UNSECEXCESS=tf.feature_column.categorical_column_with_identity('UNSECEXCESS', num_buckets=2)
    UNSECPRNULL=tf.feature_column.categorical_column_with_identity('UNSECPRNULL', num_buckets=2)
    UNSECPRVALUE=tf.feature_column.numeric_column('UNSECPRVALUE')
    UNSECPRVALUESQR=tf.feature_column.numeric_column('UNSECPRVALUESQR')
    UNSECPRVALUELOG=tf.feature_column.numeric_column('UNSECPRVALUELOG')
    AVGMNTHINULL=tf.feature_column.categorical_column_with_identity('AVGMNTHINULL', num_buckets=2)
    AVGMNTHIVALUE=tf.feature_column.numeric_column('AVGMNTHIVALUE')
    AVGMNTHIVALUESQR=tf.feature_column.numeric_column('AVGMNTHIVALUESQR')
    AVGMNTHIVALUELOG=tf.feature_column.numeric_column('AVGMNTHIVALUELOG')
    IEINDEX=tf.feature_column.numeric_column('IEINDEX')
    IEGAP=tf.feature_column.numeric_column('IEGAP')
    
    RealPropValueBuckets = tf.feature_column.bucketized_column(REALPROPVALUE, boundaries=[50000,100000,150000,200000,250000,300000,350000,400000,500000,600000])
    PersPropValueBuckets = tf.feature_column.bucketized_column(PERSPROPVALUE, boundaries=[5000,10000,15000,20000,25000,30000,35000,40000,50000,60000,70000,80000,90000,100000])
    UnsecBuckets = tf.feature_column.bucketized_column(UNSECNPRVALUE, boundaries=[5000,10000,15000,20000,25000,30000,35000,40000,50000,60000,70000,80000,90000,100000,140000])
    PrioBuckets = tf.feature_column.bucketized_column(UNSECPRVALUE, boundaries=[5000,10000,15000,20000,25000,30000,35000,40000,50000,60000])
    IncomeBuckets = tf.feature_column.bucketized_column(AVGMNTHIVALUE, boundaries=[100,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,10000,20000])
    
    # Wide columns and deep columns.  #this section is not needed to run the neural network
    base_columns = [
    #    NTRDBT,
        JOINT,
        ORGD1FPRSE,
    #    PRFILE,
        DISTSUCCESS,
    #    FEEP,
    #    FEEI,
    #    FEEW,
        REALPROPNULL,
        REALPROPNONE,
    #    REALPROPVALUE,
    #    REALPROPVALUESQR,
    #    REALPROPVALUELOG,
    #    PERSPROPNULL,
    #    PERSPROPVALUE,
    #    PERSPROPVALUESQR,
    #    PERSPROPVALUELOG,
    #    UNSECNPRNULL,
    #    UNSECNPRVALUE,
    # #   UNSECNPRVALUESQR,
    #    UNSECNPRVALUELOG,
    #    UNSECEXCESS,
    #    UNSECPRNULL,
    #    UNSECPRVALUE,
    #    UNSECPRVALUESQR,
    #    UNSECPRVALUELOG,
    #    AVGMNTHINULL,
    #    AVGMNTHIVALUE,
    #    AVGMNTHIVALUESQR,
    #    AVGMNTHIVALUELOG,
    #    IEINDEX,
        IEGAP,
        RealPropValueBuckets,
      ]

    crossed_columns = []   #no crossed columns in this implementation 

    wide_columns = base_columns + crossed_columns

    deep_columns = [    #features not used in the neural network have been commented out
        tf.feature_column.indicator_column(NTRDBT),
        tf.feature_column.indicator_column(JOINT),
        tf.feature_column.indicator_column(ORGD1FPRSE),
        tf.feature_column.indicator_column(PRFILE),
        DISTSUCCESS,
        tf.feature_column.indicator_column(FEEP),
        tf.feature_column.indicator_column(FEEI),
        tf.feature_column.indicator_column(FEEW),
        tf.feature_column.indicator_column(REALPROPNULL),
        tf.feature_column.indicator_column(REALPROPNONE),
   #     REALPROPVALUE,
   #     REALPROPVALUESQR,
   #     REALPROPVALUELOG,
        tf.feature_column.indicator_column(PERSPROPNULL),
   #     PERSPROPVALUE,
   #     PERSPROPVALUESQR,
   #     PERSPROPVALUELOG,
        tf.feature_column.indicator_column(UNSECNPRNULL),
   #     UNSECNPRVALUE,
   #     UNSECNPRVALUESQR,
   #     UNSECNPRVALUELOG,
        tf.feature_column.indicator_column(UNSECEXCESS),
        tf.feature_column.indicator_column(UNSECPRNULL),
   #     UNSECPRVALUE,
   #     UNSECPRVALUESQR,
   #     UNSECPRVALUELOG,
        tf.feature_column.indicator_column(AVGMNTHINULL),
   #     AVGMNTHIVALUE,
   #     AVGMNTHIVALUESQR,
   #     AVGMNTHIVALUELOG,
        IEINDEX,
        IEGAP,
        RealPropValueBuckets,
        PersPropValueBuckets, 
        UnsecBuckets,
        PrioBuckets,
        IncomeBuckets,
    ]

    return wide_columns, deep_columns


def build_estimator(model_dir, model_type):
    """Build an estimator appropriate for the given model type."""
    wide_columns, deep_columns = build_model_columns()
    hidden_units = [256, 128, 64, 32]

    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    run_config = tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto(device_count={'GPU': 0}))

    if model_type == 'wide':
        return tf.estimator.LinearClassifier(
        model_dir=model_dir,
        feature_columns=wide_columns, optimizer=tf.train.FtrlOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=_l1_strength,       
        l2_regularization_strength=_l2_strength),
        config=run_config)  
    
    elif model_type == 'deep':
        return tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=hidden_units,
        config=run_config)
    
    else:
        return tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config)


def input_fn(data_file, num_epochs, shuffle, batch_size):
    """Generate an input function for the Estimator."""
    assert tf.gfile.Exists(data_file), (
      '%s not found. Please make sure you have either run data_download.py or '
      'set both arguments --train_data and --test_data.' % data_file)

    def parse_csv(value):
        print('Parsing', data_file)
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('SUCCESS')            
        return features, tf.equal(labels, 1)      

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])  

    dataset = dataset.map(parse_csv, num_parallel_calls=5)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset

#set logging and parsing
tf.logging.set_verbosity(tf.logging.INFO)
FLAGS, unparsed = parser.parse_known_args()
trainAccuracyLog={}
testAccuracyLog={}

# Clean up the model directory if present
shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
model = build_estimator(FLAGS.model_dir, FLAGS.model_type)

# Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
    model.train(input_fn=lambda: input_fn(
        FLAGS.train_data, FLAGS.epochs_per_eval, True, FLAGS.batch_size))
    
    train_results = model.evaluate(input_fn=lambda: input_fn(
        FLAGS.train_data, 1, False, FLAGS.batch_size))

    test_results = model.evaluate(input_fn=lambda: input_fn(
        FLAGS.test_data, 1, False, FLAGS.batch_size))

    # Display evaluation metrics
    print('Results at epoch', (n + 1) * FLAGS.epochs_per_eval)
    print('-' * 60)
    
    if n == 24:
        for key in sorted(train_results):
            print('Training: %s: %s' % (key, train_results[key]))

    for key in sorted(test_results):
        print('Test: %s: %s' % (key, test_results[key]))
        
    trainAccuracyLog[n + 1]=train_results['accuracy']
    testAccuracyLog[n + 1]=test_results['accuracy']

#write accuracy numbers to a log file for later review    
with open("log.csv", "w", newline='') as log:
    w = csv.writer(log)
    w.writerow(['epoch','training accuracy','testing accuracy'])
    for key in trainAccuracyLog:
        w.writerow([key, trainAccuracyLog[key], testAccuracyLog[key]])
       
    

