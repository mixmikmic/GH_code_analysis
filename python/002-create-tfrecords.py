import numpy as np
import pandas as pd
import tensorflow as tf
import skimage.io as sio
import os.path as op

labels_df = pd.read_csv(op.expanduser(op.join('~', 'data_ucsf', 'deep_learning', 'labels.csv')))

labels_df.head(5)

def _int64_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_tfrecords(labels_df, fname, idx):
    writer = tf.python_io.TFRecordWriter(fname)
    for example_idx in idx:
        # Read in the data one-by-one:
        image = sio.imread(op.expanduser(op.join('~', 'data_ucsf', 'deep_learning', 'cells', 
                           labels_df['file'][example_idx])))
        label = labels_df['label'][example_idx]
        rows = image.shape[0]
        cols = image.shape[1]
        depth = image.shape[2]
        image_raw = image.tostring()
        # construct the Example proto object
        example = tf.train.Example(
            # Example contains a Features proto object
            features=tf.train.Features(feature={
            # Features contains a map of string to Feature proto objects
                'image/height': _int64_feature(rows),
                'image/width': _int64_feature(cols), 
                'image/depth': _int64_feature(depth),
                'label': _int64_feature(int(label)),
                'image/raw': _bytes_feature(image_raw)}))
                
        # use the proto object to serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized object to disk
        writer.write(serialized)

    writer.close()

idx = np.arange(labels_df.shape[0])
np.random.shuffle(idx)

prop_train = 0.6
prop_eval = 0.2 
# First 60% are for training:
train_idx = idx[:int(prop_train*idx.shape[0])]
# Next 20% are for evaluation:
eval_idx = idx[int(prop_train*idx.shape[0]):int(prop_train*idx.shape[0] + prop_eval*idx.shape[0])]
# Last 20% are for testing:
test_idx = idx[int(prop_train*idx.shape[0] + prop_eval*idx.shape[0]):]

import os.path as op
tfrecords_train_file = op.expanduser(op.join('~', 'data_ucsf', 'cells_train.tfrecords'))
tfrecords_eval_file = op.expanduser(op.join('~', 'data_ucsf', 'cells_eval.tfrecords'))
tfrecords_test_file = op.expanduser(op.join('~', 'data_ucsf', 'cells_test.tfrecords'))

write_tfrecords(labels_df, tfrecords_train_file, train_idx)

write_tfrecords(labels_df, tfrecords_eval_file, eval_idx)

write_tfrecords(labels_df, tfrecords_test_file, test_idx)



