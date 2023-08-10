from sagemaker import get_execution_role

# Bucket location to get training data
sample_data_location = 's3://jayway-robocar-raw-data/samples'

# Create a data directory
get_ipython().system('mkdir -pv ~/SageMaker/data')
get_ipython().system('aws s3 cp {sample_data_location}/ore.zip ~/SageMaker/data')

# Unzip to data dir
get_ipython().system('unzip -o ~/SageMaker/data/ore.zip -d ~/SageMaker/data')

# Check one of the records
get_ipython().system('cat ~/SageMaker/data/tub_8_18-02-09/record_3658.json | jq')

get_ipython().system('cat ~/SageMaker/data/tub_8_18-02-09/meta.json | jq')

import os
import glob
import pandas as pd
from PIL import Image

def read_tub(path):
    '''
    Read a Tub directory into memory
    
    A Tub contains records in json format, one file for each sample. With a default sample frequency of 20 Hz,
    a 5 minute drive session will contain roughly 6000 files.
    
    A record JSON object has the following properties (per default):
    - 'user/angle'      - wheel angle
    - 'user/throttle'   - speed
    - 'user/mode'       - drive mode (.e.g user or pilot)
    - 'cam/image_array' - relative path to image
    
    Returns a list of dicts, [ { 'record_id', 'angle', 'throttle', 'image', } ]
    '''

    def as_record(file):
        '''Parse a json file into a Pandas Series (vector) object'''
        return pd.read_json(file, typ='series')
    
    def is_valid(record):
        '''Only records with angle, throttle and image are valid'''
        return hasattr(record, 'user/angle') and hasattr(record, 'user/throttle') and hasattr(record, 'cam/image_array')
        
    def map_record(file, record):
        '''Map a Tub record to a dict'''
        # Force library to eager load the image and close the file pointer to prevent 'too many open files' error
        img = Image.open(os.path.join(path, record['cam/image_array']))
        img.load()
        # Strip directory and 'record_' from file name, and parse it to integer to get a good id
        record_id = int(os.path.splitext(os.path.basename(file))[0][len('record_'):])
        return {
            'record_id': record_id,
            'angle': record['user/angle'],
            'throttle': record['user/throttle'],
            'image': img
        }
    
    path = os.path.expanduser(path)
    json_files = glob.glob(os.path.join(path, '*.json'))
    records = ((file, as_record(file)) for file in json_files)
    return list(map_record(file, record) for (file, record) in records if is_valid(record))

get_ipython().run_cell_magic('time', '', "records = read_tub('~/SageMaker/data/tub_8_18-02-09')\nprint('parsed Tub into {} records'.format(len(records)))")

print(records[100])
records[100]['image']

df = pd.DataFrame.from_records(records).set_index('record_id') # Use record_id as index
df.sort_index(inplace=True)                                    # Do not create a new copy when sorting
pd.set_option('display.max_columns', 10)                       # Make sure we can see all of the columns
pd.set_option('display.max_rows', 20)                          # Keep the output on one page
df

# Displays the top 5 rows (i.e. not top 5 elements based on label index)
df.head()

# Similar to head, but displays the last rows
df.tail()

# The dimensions of the dataframe as a (rows, cols) tuple
df.shape

# The number of columns. Equal to df.shape[0]
len(df) 

# An array of the column names
df.columns 

# Columns and their types
df.dtypes

# Axes
df.axes

# Converts the frame to a two-dimensional table
df.values 

# Displays descriptive stats for all columns
df.describe()

# Select one element returns a Pandas.Series object
df.loc[1]

# Select multiple elements returns a Pandas.DataFrame object
df.loc[1:5]

# Plot throttle only
get_ipython().run_line_magic('matplotlib', 'inline')

df.plot.line(y='throttle')

# Plot both throttle and angle next to each other
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=2)
df.plot(ax=axes[0], kind='line', y='throttle', color='orange')
df.plot(ax=axes[1], kind='density', y='angle', color='red')
plt.figure()

get_ipython().run_cell_magic('time', '', "import numpy\nfrom cv2 import VideoWriter, VideoWriter_fourcc, cvtColor, COLOR_RGB2BGR\nfrom contextlib import contextmanager\n\n@contextmanager\ndef VideoCreator(*args, **kwargs):\n    v = VideoWriter(*args, **kwargs)\n    try:\n        yield v\n    finally:\n        v.release()\n\ndef make_video(images, out='donkey-run.mp4', fps=20):\n    '''\n    Creates a video from PIL images\n    '''\n    if (len(images) <= 0):\n      raise ValueError('Images array must not be empty')\n    \n    # Extract size from first image\n    size = images[1].size\n    \n    # Create codec\n    fourcc = VideoWriter_fourcc(*'H264')\n    \n    # Create a VideoCreator and return the new video\n    with VideoCreator(out, fourcc, float(fps), size) as v:\n        for img in images:\n            arr = cvtColor(numpy.array(img), COLOR_RGB2BGR)\n            v.write(arr)\n\n    return out\n\nvideo_file = os.path.join(os.path.expanduser('~/SageMaker'), 'donkey-run.mp4')\nmake_video(df['image'], video_file)\nprint(video_file)")

ls -la /home/ec2-user/SageMaker/donkey-run.mp4

