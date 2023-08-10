# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

# Config the matplotlib backend as plotting inline in IPython
get_ipython().magic('matplotlib inline')

url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None

def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 1% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
      
    last_percent_reported = percent
        
def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  if force or not os.path.exists(filename):
    print('Attempting to download:', filename) 
    filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

num_classes = 10
np.random.seed(133)

def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall()
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders
  
train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)

from IPython.display import Image
Image(filename='notMNIST_large/A/VXBkaWtlLnR0Zg==.png') 

Image(filename='notMNIST_large/A/Q29zbW9zLU1lZGl1bS5vdGY=.png') 

Image(filename='notMNIST_small/A/RGF5dHJpcHBlciBQbGFpbi50dGY=.png') 

Image(filename='notMNIST_small/A/SHVtYW5pc3QgOTcwIEJvbGQucGZi.png') 

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)

train_datasets[:]

type(train_datasets)

test_datasets[:]

ex = pickle.load( open( "notMNIST_small/A.pickle", "rb" ) )
ex.shape

plt.imshow(ex[1,:,:])

plt.imshow(ex[2,:,:])

plt.imshow(ex[3,:,:])

train_freq = np.zeros(10)
test_freq = np.zeros(10)

prefs = ['A' , 'B' , 'C' , 'D' , 'E' , 'F' , 'G', 'H', 'I', 'J']
i = 0
for pref in prefs:
    tr = pickle.load( open( "notMNIST_large/"+pref+".pickle", "rb" ) )
    ts = pickle.load( open( "notMNIST_small/"+pref+".pickle", "rb" ) )
    train_freq[i] = tr.shape[0]
    test_freq[i] = ts.shape[0]
    i = i + 1

print("***train_freq****")
print(train_freq)
print(train_freq/np.sum(train_freq))
print("\n***test_freq****")
print(test_freq)
print(test_freq/np.sum(test_freq))

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                    
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels
            
            
train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

train_dataset.shape

plt.imshow(train_dataset[3,:,:])

plt.imshow(test_dataset[3,:,:])

plt.imshow(valid_dataset[3,:,:])

pickle_file = 'notMNIST.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
    
statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

oTrVal = np.zeros(200) ## sample 
oTrTest = np.zeros(200) ## sample

for i in range(0,oTrVal.shape[0]):
    if (i % 100 == 0):
        sys.stdout.write("..%s" % i)
    for j in range(0,train_dataset.shape[0]):
        #if np.array_equal(train_dataset[j,:,:],valid_dataset[i,:,:]):
        if np.sum(np.subtract(train_dataset[j,:,:],valid_dataset[i,:,:]))==0:
            oTrVal[i] = 1 
            break 
print("\n***Xval**")
print(np.sum(oTrVal)/oTrVal.shape[0])

for i in range(0,oTrTest.shape[0]):
    if (i % 100 == 0):
        sys.stdout.write("..%s" % i)
    for j in range(0,train_dataset.shape[0]):
        #if np.array_equal(train_dataset[j,:,:],valid_dataset[i,:,:]):
        if np.sum(np.subtract(train_dataset[j,:,:],test_dataset[i,:,:]))==0:
            oTrTest[i] = 1 
            break 
print("\n***XTest**")
print(np.sum(oTrTest)/oTrTest.shape[0])

from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression

train_dataset_100 = train_dataset[0:100,:,:]
train_dataset_100 = np.reshape(train_dataset_100,(100,784))
train_labels_100 = train_labels[0:100]

clf = GridSearchCV(LogisticRegression(penalty='l2'), 
                   scoring ='accuracy',
                   param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]})
clf = clf.fit( train_dataset_100, train_labels_100) 
print(">>> Best accuracy:"+str(clf.best_score_))
print(">>> Best Params:"+str(clf.best_params_))

train_dataset_1000 = train_dataset[0:1000,:,:]
train_dataset_1000 = np.reshape(train_dataset_1000,(1000,784))
train_labels_1000 = train_labels[0:1000]

clf = GridSearchCV(LogisticRegression(penalty='l2'), 
                   scoring ='accuracy',
                   param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]})
clf = clf.fit( train_dataset_1000, train_labels_1000) 

print(">>> Best accuracy:"+str(clf.best_score_))
print(">>> Best Params:"+str(clf.best_params_))

train_dataset_5000 = train_dataset[0:5000,:,:]
train_dataset_5000 = np.reshape(train_dataset_5000,(5000,784))
train_labels_5000 = train_labels[0:5000]

clf = GridSearchCV(LogisticRegression(penalty='l2'), 
                   scoring ='accuracy',
                   param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]})
clf = clf.fit( train_dataset_5000, train_labels_5000) 

print(">>> Best accuracy:"+str(clf.best_score_))
print(">>> Best Params:"+str(clf.best_params_))

from sklearn.ensemble import RandomForestClassifier
    
train_dataset_5000 = train_dataset[0:5000,:,:]
train_dataset_5000 = np.reshape(train_dataset_5000,(5000,784))
train_labels_5000 = train_labels[0:5000]

clf = GridSearchCV(RandomForestClassifier( n_estimators = 1000 ), 
                   scoring ='accuracy',param_grid={})
clf = clf.fit( train_dataset_5000, train_labels_5000) 

print(">>> Best accuracy:"+str(clf.best_score_))
print(">>> Best Params:"+str(clf.best_params_))

