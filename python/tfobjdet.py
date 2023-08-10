import os
import pandas as pd

#import autti labels to dataframe

annotation_root = os.path.join("annotations", "object-dataset")
annotation_path = os.path.join(annotation_root, "labels.csv")

df = pd.read_csv(annotation_path, sep = " ", header = None, names = ["frame", "xmin", "ymin", "xmax", "ymax", "occluded", "label","attributes"])

df.head()

#only take car labels
cardf = df[ df.label == 'car']

cardf = cardf.drop(['label', 'attributes'], 1)

cardf.head()

#group by frames and convert object detections to list
cardfg = cardf.groupby(['frame'], as_index = False)

cardfl = cardfg.aggregate( lambda x : list(x) )

cardfl.head()

# Add relative path to files
cardfl.reset_index()
cardfl['frame'] = cardfl['frame'].apply(lambda x: os.path.join(annotation_root, x))
cardfl.head()

import tensorflow as tf
from object_detection.utils import dataset_util
import hashlib
import io
import PIL.Image

def create_tf_example(df_row):
  #Populate the following variables from your example.
  filename = df_row['frame']
  
  with tf.gfile.GFile(filename, 'rb') as fid:
    encoded_jpg = fid.read()
  
  encoded_image_data = io.BytesIO(encoded_jpg) # Encoded image bytes
  image = PIL.Image.open(encoded_image_data)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  height = 1200 # Image height
  width = 1920 # Image width
  
  image_format = b'jpeg'

  xmins = [max(x / width, 0) for x in df_row['xmin'] ] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [min(x / width, 1) for x in df_row['xmax'] ] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [max(y / height, 0) for y in df_row['ymin'] ] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [min(y / height, 1) for y in df_row['ymax'] ] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = ["car".encode('utf8')] * len(xmins) # List of string class name of bounding box (1 per box)
  classes = [1] * len(xmins) # List of integer class id of bounding box (1 per box)
    
  fnamebytes = filename.encode()

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(fnamebytes),
      'image/source_id': dataset_util.bytes_feature(fnamebytes),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example

#read some images and display them
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')
cardfl.iloc[2]['frame']

import cv2
f, axs = plt.subplots(3, 3, figsize = (20, 10))
for ii in range(3):
    for jj in range(3):
        inx = ii * 3 + jj + 1000
        filename = cardfl.iloc[inx]['frame']
        xmin = cardfl.iloc[inx]['xmin']
        ymin = cardfl.iloc[inx]['ymin']
        xmax = cardfl.iloc[inx]['xmax']
        ymax = cardfl.iloc[inx]['ymax']
        img = mpimg.imread(filename)
        for xm,ym,xma,yma in zip(xmin, ymin,xmax,ymax):
            cv2.rectangle(img, (xm, ym), (xma,yma), color = (0,255,0), thickness = 2)
        if ii == 0 and jj == 0:
            print(img.shape)
        axs[ii, jj].imshow(img)
plt.tight_layout()

#do the same for the crowdai dataset
annotation_path = "labels_crowdai.csv"

df = pd.read_csv(annotation_path, sep = ",")

print(df.columns)
df = df.drop(df.columns[ [-1] ], axis = 1)
df.head()

#reorder columns
df = df[['Frame','xmin', 'ymin','xmax','ymax','Label']]
df.rename(columns={'Frame': 'frame', 'Label': 'label'}, inplace=True)
df.head()

#group by frames and convert object detections to list
cardfg_1 = df.groupby(['frame'], as_index = False)

cardfl_1 = cardfg_1.aggregate( lambda x : list(x) )

# Add relative path to files
cardfl_1.reset_index()
cardfl_1['frame'] = cardfl_1['frame'].apply(lambda x: os.path.join("annotations", "object-detection-crowdai", x))
cardfl_1.head()

#visualize dataset
f, axs = plt.subplots(3, 3, figsize = (15, 10))
for ii in range(3):
    for jj in range(3):
        inx = ii * 3 + jj + 100
        filename = cardfl_1.iloc[inx]['frame']
        xmin = cardfl_1.iloc[inx]['xmin']
        ymin = cardfl_1.iloc[inx]['ymin']
        xmax = cardfl_1.iloc[inx]['xmax']
        ymax = cardfl_1.iloc[inx]['ymax']
        img = mpimg.imread(filename)
        for xm,ym,xma,yma in zip(xmin, ymin,xmax,ymax):
            cv2.rectangle(img, (xm, ym), (xma,yma), color = (0,255,0), thickness = 2)
        if ii == 0 and jj == 0:
            print(img.shape)
        axs[ii, jj].imshow(img)
plt.tight_layout()

from sklearn.model_selection import train_test_split

def write_records(output_path, df):
  writer = tf.python_io.TFRecordWriter(output_path)

  for _ , row in df.iterrows():
    tf_example = create_tf_example(row)
    writer.write(tf_example.SerializeToString())

  writer.close()

#combine both datasets
car_df_combined = pd.concat( [cardfl, cardfl_1] )

#split dataset into training and test sets
train, test = train_test_split(cardfl, test_size = 0.2)

#write records

write_records('objdetdata/train.record', train)
write_records('objdetdata/test.record', test)



