from pathlib import Path
import numpy as np

# Image sizes
MAX_WIDTH = 1280
MAX_HEIGHT = 300

# Training params
TRAIN_TEST_SPLIT = 0.9

# Class dictionary
CLASS_NAMES = {1:'cloud', 2:'sun', 3:'house', 4:'tree'}

# Define paths to sub-folders
root_dir = Path.cwd()
images_path = root_dir / 'images'
labels_path = root_dir / 'labels'
train_path = root_dir / 'models'
data_path = root_dir / 'data'

# Output filenames paths
train_tfrecord_path = data_path / 'train.record'
test_tfrecord_path = data_path / 'test.record'
labels_csv_path = data_path / 'labels.csv'

import pandas as pd
import xml.etree.ElementTree as ET

# Convert the XMLs into a single CSV file
xml_list = []
for xml_path in list(labels_path.glob('*.xml')):
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    for member in root.findall('object'):
        # Unpack each object (BB) from the xml
        value = (root.find('filename').text,
                 int(root.find('size')[0].text),
                 int(root.find('size')[1].text),
                 member[0].text,
                 int(member[4][0].text),
                 int(member[4][1].text),
                 int(member[4][2].text),
                 int(member[4][3].text))
        xml_list.append(value)
# Create pandas dataframe from the labels in the XML
column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
xml_df = pd.DataFrame(xml_list, columns=column_name)
xml_df.to_csv(str(labels_csv_path), index=None)
print('Converted xmls to csv file')

import io
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util

def to_tfrecords(image_paths, labels_path, tfrecord_path):
    if tfrecord_path.exists():
        print('TFRecord already created, delete it before making a new one')
        return
    writer = tf.python_io.TFRecordWriter(str(tfrecord_path))
    # Read labels from csv
    label_df = pd.read_csv(str(labels_path))
    gb = label_df.groupby('filename')
    # Convert each image to a tfrecords example then write
    for image_path in image_paths:
        try:
            group = gb.get_group(image_path.name)
        except KeyError:
            print('Could not find labels for %s' % image_path.name)
            continue
        # Write each serialized example to writer
        writer.write(_create_tf_example(image_path, group).SerializeToString())
    writer.close()
    print('TFRecord created at %s' % str(tfrecord_path))

def _create_tf_example(image_path, groups):
    # Read image and encode it
    with tf.gfile.GFile(str(image_path), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    # Feature defines each discrete entry in the tfrecords file
    filename = image_path.name.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    print('groups: ', groups)
    for index, row in groups.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes.append(row['class'])
        classes_text.append(CLASS_NAMES[row['class']].encode('utf8'))
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return example

# Split data into test and train
image_paths = list(images_path.glob('*.jpg'))
num_images = len(image_paths)
num_train = int(TRAIN_TEST_SPLIT * num_images)
train_index = np.random.choice(num_images, size=num_train, replace=False)
test_index = np.setdiff1d(list(range(num_images)), train_index)
train_image_paths = [image_paths[i] for i in train_index]
test_image_paths = [image_paths[i] for i in test_index]
print('There are %d images total, split into %s train and %s test' % (num_images,
                                                                      len(train_image_paths),
                                                                      len(test_image_paths)))
# Convert list of train and test images into a tfrecord
to_tfrecords(train_image_paths, labels_csv_path, train_tfrecord_path)
to_tfrecords(test_image_paths, labels_csv_path, test_tfrecord_path)



