from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
from IPython.display import display, Image
from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
get_ipython().magic('matplotlib inline')

url = 'http://ufldl.stanford.edu/housenumbers/'
import urllib    

tar_data_list=['train.tar.gz','test.tar.gz','extra.tar.gz']
for sample_data in tar_data_list:
    print ("Size of "+sample_data +" : ",urllib.urlopen(url+sample_data).info()['Content-Length'])   

def maybe_download(filename, expected_bytes):
#   """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urllib.urlretrieve(url + filename, filename)
  statinfo_size = os.stat(filename).st_size
  if statinfo_size == expected_bytes:
    print ('Found and verified', filename)
  else:
    raise Exception(
      'Failed to verify' + filename + '. Can you get to it with a browser?')
  return filename
    
train_filename = maybe_download('train.tar.gz',404141560)  #TODO
test_filename = maybe_download('test.tar.gz',276555967)  #TODO
extra_filename = maybe_download('extra.tar.gz',1955489752)

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
  data_folders = root
  print(data_folders)
  return data_folders
  
train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)
extra_folders = maybe_extract(extra_filename)

#!pip install h5py

#Reference from udacity Forum: https://discussions.udacity.com/t/how-to-deal-with-mat-files/160657
import h5py

# The DigitStructFile is just a wrapper around the h5py data.  It basically references 
#    inf:              The input h5 matlab file
#    digitStructName   The h5 ref to all the file names
#    digitStructBbox   The h5 ref to all struc data
class DigitStructFile:
    def __init__(self, inf):
        self.inf = h5py.File(inf, 'r')
        self.digitStructName = self.inf['digitStruct']['name']
        self.digitStructBbox = self.inf['digitStruct']['bbox']

# getName returns the 'name' string for for the n(th) digitStruct. 
    def getName(self,n):
        return ''.join([chr(c[0]) for c in self.inf[self.digitStructName[n][0]].value])

# bboxHelper handles the coding difference when there is exactly one bbox or an array of bbox. 
    def bboxHelper(self,attr):
        if (len(attr) > 1):
            attr = [self.inf[attr.value[j].item()].value[0][0] for j in range(len(attr))]
        else:
            attr = [attr.value[0][0]]
        return attr

# getBbox returns a dict of data for the n(th) bbox. 
    def getBbox(self,n):
        bbox = {}
        bb = self.digitStructBbox[n].item()
        bbox['height'] = self.bboxHelper(self.inf[bb]["height"])
        bbox['label'] = self.bboxHelper(self.inf[bb]["label"])
        bbox['left'] = self.bboxHelper(self.inf[bb]["left"])
        bbox['top'] = self.bboxHelper(self.inf[bb]["top"])
        bbox['width'] = self.bboxHelper(self.inf[bb]["width"])
        return bbox

    def getDigitStructure(self,n):
        s = self.getBbox(n)
        s['name']=self.getName(n)
        return s

# getAllDigitStructure returns all the digitStruct from the input file.     
    def getAllDigitStructure(self):
        return [self.getDigitStructure(i) for i in range(len(self.digitStructName))]

# Return a restructured version of the dataset (one structure by boxed digit).
#
#   Return a list of such dicts :
#      'filename' : filename of the samples
#      'boxes' : list of such dicts (one by digit) :
#          'label' : 1 to 9 corresponding digits. 10 for digit '0' in image.
#          'left', 'top' : position of bounding box
#          'width', 'height' : dimension of bounding box
#
# Note: We may turn this to a generator, if memory issues arise.
    def getAllDigitStructure_ByDigit(self):
        pictDat = self.getAllDigitStructure()
        result = []
        structCnt = 1
        for i in range(len(pictDat)):
            item = { 'filename' : pictDat[i]["name"] }
            figures = []
            for j in range(len(pictDat[i]['height'])):
               figure = {}
               figure['height'] = pictDat[i]['height'][j]
               figure['label']  = pictDat[i]['label'][j]
               figure['left']   = pictDat[i]['left'][j]
               figure['top']    = pictDat[i]['top'][j]
               figure['width']  = pictDat[i]['width'][j]
               figures.append(figure)
            structCnt = structCnt + 1
            item['boxes'] = figures
            result.append(item)
        return result

train_folders = 'train'
test_folders = 'test'
extra_folders = 'extra'
#train_data creation from digitSturct.mat
fin = os.path.join(train_folders, 'digitStruct.mat')
dsf = DigitStructFile(fin)
train_data = dsf.getAllDigitStructure_ByDigit()

print (type(train_data))   #type is list

#test for  the purpose  of the analysis of dataset.
print("boxes, filename:",len(train_data[0]))
print("boxes, filename:",len(train_data[1]))
#print out  train_data from element0 to element1(2 dataset)
train_data[0:2]

#test
print(len(train_data))
print(train_data[0])
print("*"*50)
print("train_data[0]['boxes']:\n",train_data[0]['boxes'])
print("*"*20,"same following two (dictionary) **************")
print("train_data[0]['boxes'][0]:\n",train_data[0]['boxes'][0])
print("train_data[0]['boxes'][1]:\n",train_data[0]['boxes'][1])
print("train_data[0]['boxes'][1]['width']:\n",train_data[0]['boxes'][1]['width'])
print("*"*30)
for e in train_data[0]['boxes'][1].keys():
    print (e)
for e in train_data[0]['boxes'][1].values():
    print (e)
#print(train_data[0][[1,2,3,4]])
#print(extra_data[0][[1,2,3,4]])

#test

my_list=[]
for i in xrange(6):
    my_list.append(train_data[i]['boxes'][1]['label'])
print(my_list)   
print("*"*50)
for i in xrange(6):
    print(train_data[i]['boxes'][0]['label'])

#test 
my_list1=[]
for i in xrange(2):
#     for j in xrange(5):
        my_list1.append(train_data[0]['boxes'][i]['label'])
print(my_list1)        
print("train_data[0]['filename']:",train_data[0]['filename'])

#test 1stt element of train_data structure 
train_data[0] #print(len(train_data[0])) : output is 2

#test_data creation from digitSturct.mat
fin = os.path.join(test_folders, 'digitStruct.mat')
dsf = DigitStructFile(fin)
test_data = dsf.getAllDigitStructure_ByDigit()

#test for test_data,  first element's contents
test_data[0]

#extra_data creation from digitSturct.mat
fin = os.path.join(extra_folders, 'digitStruct.mat')
dsf = DigitStructFile(fin)
extra_data = dsf.getAllDigitStructure_ByDigit()

#!pip install image

from PIL import Image
train_imsize = np.ndarray([len(train_data),2])  #random num:(33402,2)
for i in np.arange(len(train_data)): #33402
    filename = train_data[i]['filename']  #train_data[0]['filename']:1.png
    fullname = os.path.join(train_folders, filename)
    im = Image.open(fullname)
    train_imsize[i, :] = im.size[:] 

print("max num along column 0:",np.amax(train_imsize[:,0]), 
      ", max num along column 1:",np.amax(train_imsize[:,1]))
print("min num along column 0:",np.amin(train_imsize[:,0]),
       ", min num along column 1:",np.amin(train_imsize[:,1]))

#test
from PIL import Image
print("train_imsize = np.ndarray([len(train_data),2])::(33402,2):\n",
      np.ndarray([len(train_data),2])) #(33402,2)
i=0
f1=train_data[i]['filename']
print("train_data[i]['filename']:",f1)
full1=os.path.join(train_folders, f1)
print("full1 name of image file:",full1)
im=Image.open(fullname)
print(im)
train_imsize[i, :] = im.size[:] #random num:(33402,2) from above  cell
print("train_imsize[0, :]:",train_imsize[i, :])
print("len(train_imsize[:,0]):",len(train_imsize[:, 0]))
print("train_imsize[:3,0]:",train_imsize[:3, 0])
print("len(train_imsize[:,1]):",len(train_imsize[:, 1]))
print("train_imsize[:3,1]:",train_imsize[:3, 1])

test_imsize = np.ndarray([len(test_data),2]) #random num:(33402,2)
for i in np.arange(len(test_data)): #33402
    filename = test_data[i]['filename'] #train_data[0]['filename']:1.png
    fullname = os.path.join(test_folders, filename)
    im = Image.open(fullname)
    test_imsize[i, :] = im.size[:]
#maximum number among all elements in column 0 and maximum number in column 1
print("max num along column 0:",np.amax(test_imsize[:,0]), 
      ", max num along column 1:",np.amax(test_imsize[:,1]))
#minimum number among all elements in column 0 and minimum number in column 1
print("min num along column 0:",np.amin(test_imsize[:,0]), 
      ", min num along column 1:",np.amin(test_imsize[:,1]))

extra_imsize = np.ndarray([len(extra_data),2])
for i in np.arange(len(extra_data)):
    filename = extra_data[i]['filename']
    fullname = os.path.join(extra_folders, filename)
    im = Image.open(fullname)
    extra_imsize[i, :] = im.size[:]
#maximum number among all elements in column 0 and maximum number in column 1
print("max num along column 0:",np.amax(extra_imsize[:,0]), 
      ", max num along column 1:",np.amax(extra_imsize[:,1]))
#minimum number among all elements in column 0 and minimum number in column 1
print("min num along column 0:",np.amin(extra_imsize[:,0]), 
      ", min num along column 1:",np.amin(extra_imsize[:,1]))

#test
print(np.where(train_imsize[:,0]==876))
print(np.where(train_imsize[:,0]==25))
print(np.where(test_imsize[:,0]==1083))
print(np.where(test_imsize[:,0]==31))
print(np.where(extra_imsize[:,0]==668))
print(np.where(extra_imsize[:,0]==22))

#test for train_data contnets
print("len(train_data[0]['boxes']):",len(train_data[0]['boxes']))
print("train_data[0]:\n",train_data[0])
print("len(train_data[4440]['boxes']):",len(train_data[4440]['boxes']))
print("train_data[4440]:\n",train_data[4440])
for i in xrange(3):
    print(len(train_data[i]['boxes'])) # now 2 but  it can be 1,2,3,4
print(np.ndarray([len(train_data[i]['boxes'])], dtype='float32'))#np.ndarray([2],dtype=f)

from collections import Counter
digit_length_list=[]
for  i in xrange(len(train_data)): #33402
    digit_length_list.append(len(train_data[i]['boxes']))  
digit_count=Counter(digit_length_list)  
print("Number of character:corresponding frequncy - ",digit_count) 
l = plt.bar(digit_count.keys(), digit_count.values(), 1, color="blue")

plt.xlabel('Number of characters')
plt.ylabel('Sample Size')
plt.grid(True)
plt.title('Training Dataset')

plt.show()

from collections import Counter
digit_length_list=[]
for  i in xrange(len(test_data)): #33402
    digit_length_list.append(len(test_data[i]['boxes']))   
digit_count=Counter(digit_length_list)  
print("Number of character:corresponding frequncy - ",digit_count) 
l = plt.bar(digit_count.keys(), digit_count.values(), 1, color="green")

plt.xlabel('Number of characters')
plt.ylabel('Sample Size')
plt.grid(True)
plt.title('Test Dataset')

plt.show()

from collections import Counter
digit_length_list=[]
for  i in xrange(len(extra_data)): #33402
    digit_length_list.append(len(extra_data[i]['boxes']))   
digit_count=Counter(digit_length_list) 
print("Number of character:corresponding frequncy - ",digit_count) 
l = plt.bar(digit_count.keys(), digit_count.values(), 1, color="purple")

plt.xlabel('Number of characters')
plt.ylabel('Sample Size')
plt.grid(True)
plt.title('Extra Dataset')

plt.show()

import PIL.Image as Image
#generate dataset to fit for training and specific weights are applied to convet  to grayscale. 
def generate_dataset(data, folder):
    
    #len(train_data):33402,len(test_data):13068,len(extra):202353
    dataset = np.ndarray([len(data),32,32,1], dtype='float32') #train (33402,32.32,1)
    labels = np.ones([len(data),6], dtype=int) * 10 #train: (33402,6)
    #for  loop : i element, j boxes in i element.
    for i in np.arange(len(data)):
        filename = data[i]['filename']
        fullname = os.path.join(folder, filename)
        im = Image.open(fullname)
        boxes = data[i]['boxes']
        num_digit = len(boxes)  #i=0, len(boxes) =2, it  can be 1,2,3,4
        #initialize with number of boxes or digit  number.
        labels[i,0] = num_digit #now:labels[0,0]=2,next loop: labels[1,0]=2, labels[2,0]=2
        top = np.ndarray([num_digit], dtype='float32') #ndarray([2], dtype='float32')
        left = np.ndarray([num_digit], dtype='float32')
        height = np.ndarray([num_digit], dtype='float32')
        width = np.ndarray([num_digit], dtype='float32')
        for j in np.arange(num_digit):  #i=0, len(boxes) =2, it  can be 1,2,3,4
            if j < 5:  
    #From above train_data of test cell. Value of each train_data is as follow.         
    #i=0,j=0:labels[0,1]=boxes[0]['label']=data[0]['boxes'][0]['label']=1.0
    #i=0,j=1:labels[0,2]=boxes[1]['label']=data[0]['boxes'][1]['label']=9.0
    #i=4440,j=0:labels[4440,1]=boxes[0]['label']=data[4440]['boxes'][0]['label']=1.0
    #i=4440,j=1:labels[4440,2]=boxes[1]['label']=data[4440]['boxes'][1]['label']=2.0
    #i=4440,j=2:labels[4440,3]=boxes[2]['label']=data[4440]['boxes'][2]['label']=4.0
                labels[i,j+1] = boxes[j]['label'] 
                if boxes[j]['label'] == 10: labels[i,j+1] = 0  
            else: print('#',i,'image has more than 5 digits.')
    #i=0,j=0: top[0] = boxes[0]['top'] =77.0
    #i=0,j=1: top[0] = boxes[1]['top'] =81.0           
            top[j] = boxes[j]['top']
            left[j] = boxes[j]['left']
            height[j] = boxes[j]['height']
            width[j] = boxes[j]['width']
        
        im_top = np.amin(top)    #minimum value among list top
        im_left = np.amin(left)  #minimum value among list left
        im_height = np.amax(top) + height[np.argmax(top)] - im_top
        im_width = np.amax(left) + width[np.argmax(left)] - im_left
        
        im_top = np.floor(im_top - 0.1 * im_height)
        im_left = np.floor(im_left - 0.1 * im_width)
        im_bottom = np.amin([np.ceil(im_top + 1.2 * im_height), im.size[1]])
        im_right = np.amin([np.ceil(im_left + 1.2 * im_width), im.size[0]])
        #To convert from RGB to grayscale, weights(0.2989, 0.5870, 0.1140) out to be applied to channels R, G, and B.
        im = im.crop((im_left, im_top, im_right, im_bottom)).resize([32,32], Image.ANTIALIAS)
        im = np.dot(np.array(im, dtype='float32'), [[0.2989],[0.5870],[0.1140]])
        mean = np.mean(im, dtype='float32')
        std = np.std(im, dtype='float32', ddof=1)
        if std < 1e-4: std = 1. 
        im = (im - mean) / std
        dataset[i,:,:,:] = im[:,:,:]

    return dataset, labels

train_dataset, train_labels = generate_dataset(train_data, train_folders)
print(train_dataset.shape, train_labels.shape)

test_dataset, test_labels = generate_dataset(test_data, test_folders)
print(test_dataset.shape, test_labels.shape)

extra_dataset, extra_labels = generate_dataset(extra_data, extra_folders)
print(extra_dataset.shape, extra_labels.shape)

train_dataset = np.delete(train_dataset, 29929, axis=0)
train_labels = np.delete(train_labels, 29929, axis=0)

print(train_dataset.shape, train_labels.shape)

import random

random.seed()

n_labels = 10
valid_index = []
valid_index2 = []
train_index = []
train_index2 = []
for i in np.arange(n_labels):
    valid_index.extend(np.where(train_labels[:,1] == (i))[0][:400].tolist())
    train_index.extend(np.where(train_labels[:,1] == (i))[0][400:].tolist())
    valid_index2.extend(np.where(extra_labels[:,1] == (i))[0][:200].tolist())
    train_index2.extend(np.where(extra_labels[:,1] == (i))[0][200:].tolist())

random.shuffle(valid_index)
random.shuffle(train_index)
random.shuffle(valid_index2)
random.shuffle(train_index2)

valid_dataset = np.concatenate((extra_dataset[valid_index2,:,:,:], train_dataset[valid_index,:,:,:]), axis=0)
valid_labels = np.concatenate((extra_labels[valid_index2,:], train_labels[valid_index,:]), axis=0)
train_dataset_t = np.concatenate((extra_dataset[train_index2,:,:,:], train_dataset[train_index,:,:,:]), axis=0)
train_labels_t = np.concatenate((extra_labels[train_index2,:], train_labels[train_index,:]), axis=0)

print(train_dataset_t.shape, train_labels_t.shape)
print(test_dataset.shape, test_labels.shape)
print(valid_dataset.shape, valid_labels.shape)

pickle_file = 'SVHN_multi.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset_t,
    'train_labels': train_labels_t,
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

