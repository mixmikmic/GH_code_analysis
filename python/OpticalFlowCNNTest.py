import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import lmdb

get_ipython().magic('matplotlib inline')
from PIL import Image

import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

# net = caffe.Net('demoDeploy.prototxt', caffe.TEST)
import os
print(os.getcwd())

caffe_root = '../..' # assuming this notebook is in {caffe_root}/examples/optical_flow
if os.path.isfile(caffe_root + '/examples/optical_flow/opt_flow_quick_iter_2000.caffemodel'):
    print 'Caffe Model found.'
else:
    print 'Caffe Model Not Found.'
    

# http://research.beenfrog.com/code/2015/03/28/read-leveldb-lmdb-for-caffe-with-python.html
def get_data_for_id_from_lmdb(lmdb_name, id):
    lmdb_env = lmdb.open(lmdb_name, readonly=True)
    lmdb_txn = lmdb_env.begin()
    
    lmdb_cursor = lmdb_txn.cursor()
    raw_datum = lmdb_txn.get(id)
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(raw_datum)
    
    if(len(datum.data)):
        feature = np.fromstring(datum.data, dtype=float).reshape(datum.channels, datum.height, datum.width)
    label = datum.label
    
    return(label, feature)

# http://research.beenfrog.com/code/2015/03/28/read-leveldb-lmdb-for-caffe-with-python.html
def get_min_from_lmdb(lmdb_name):
    lmdb_env = lmdb.open(lmdb_name, readonly=True)
    lmdb_txn = lmdb_env.begin()
    
    lmdb_cursor = lmdb_txn.cursor()
    #raw_datum = lmdb_txn.get(id)
    datum = caffe.proto.caffe_pb2.Datum()
    #datum.ParseFromString(raw_datum)
    i=0
    for key, val in lmdb_cursor:
        print(key)
        datum.ParseFromString(val)
        if(len(datum.data)):
            #feature = np.fromstring(datum.data, dtype=float).reshape(datum.channels, datum.height, datum.width)
            feature = np.fromstring(datum.data, dtype="uint8").reshape(datum.channels, datum.height, datum.width)
        label = datum.label
        print(datum.label)
        i=i+1
        # return the ith frame
        if i==808:
            return feature
        #break
        
#    if(len(datum.data)):
#        feature = np.fromstring(datum.data, dtype=float).reshape(datum.channels, datum.height, datum.width)
#    label = datum.label
    
#    return(label, feature)

f = get_min_from_lmdb("test_bgr_flow_lmdb")
f.shape

np.save("boxingvideo77", f)

print(f.sum())
print(f[0,...])
print(np.std(f))
print(np.mean(f))

import matplotlib.mlab as mlab
n, bins, patches = plt.hist(f[1,...].flatten(), 50, normed=1, facecolor='green', alpha=0.75)

# add a 'best fit' line
y = mlab.normpdf( bins, np.mean(f[1,...]), np.std(f[1,...]))
l = plt.plot(bins, y, 'r--', linewidth=1)

plt.xlabel('')
plt.ylabel('Probability')
plt.title(r'$\mathrm{Histogram\ of\ Feature[0]:}\ \mu=mu,\ \sigma=sigma$')
plt.axis([-5, 5, 0, 0.03])
plt.grid(True)

plt.show()

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

f = get_min_from_lmdb("train_opt_flow_lmdb")
f = np.rollaxis(np.rollaxis(f, 2), 2)
print(f.shape)
hsv = np.zeros((120,160,3), dtype='uint8')
print(hsv.shape)
hsv[...,1] = 255
mag, ang = cv2.cartToPolar(f[...,0], f[...,1])
hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
hsv[...,0] = (ang*180/np.pi/2)
hsv[...,0]

# Farneback params
#flow = cv2.calcOpticalFlowFarneback(prev_gray,gray,0.5,1,3,15,3,5,1)

#np.max(hsv[...,0].astype(int))
#hsv = np.zeros((120,160,3), dtype='uint8')
bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
#hsv[...,0].dtype

#cv2.imshow("out", bgr)
np.save("myfile.txt", bgr)

temp = np.load("myfile.txt.npy")

temp.dtype
# The flow was visualized with
#bgr = np.load("myfile.txt.npy")
#cv2.imshow("demo", bgr)
#

# receive a (2, 120, 160) OF matrix and return the (3, 120, 160) BGR matrix
def OF_matrix_to_bgr(f):
    f = np.rollaxis(np.rollaxis(f, 2), 2)
    #print(f.shape)
    hsv = np.zeros((120,160,3), dtype='uint8')
    #print(hsv.shape)
    hsv[...,1] = 255
    mag, ang = cv2.cartToPolar(f[...,0], f[...,1])
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    hsv[...,0] = (ang*180/np.pi/2)
    # getting a (120, 160, 3) dimension bgr matrix
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    # rollaxis to get (3, 120, 160) dimension bgr matrix
    bgr = np.rollaxis(bgr, 2)
    return bgr

def convert_to_bgr_lmdb(lmdb_name):
    lmdb_env = lmdb.open(lmdb_name, readonly=True)
    lmdb_txn = lmdb_env.begin()
    
    lmdb_cursor = lmdb_txn.cursor()
    #raw_datum = lmdb_txn.get(id)
    datum = caffe.proto.caffe_pb2.Datum()
    #datum.ParseFromString(raw_datum)
    i=0
    for key, val in lmdb_cursor:
        print(key)
        datum.ParseFromString(val)
        if(len(datum.data)):
            feature = np.fromstring(datum.data, dtype=float).reshape(datum.channels, datum.height, datum.width)
        else:
            continue
        label = datum.label
        bgr = OF_matrix_to_bgr(feature)
        #print(bgr.shape)
        #print(np.sum(bgr[0,...]))
        #print(datum.label)
        i=i+1
        # return the ith frame
        if i==2:
            break
        
#    if(len(datum.data)):
#        feature = np.fromstring(datum.data, dtype=float).reshape(datum.channels, datum.height, datum.width)
#    label = datum.label
    
#    return(label, feature)

#convert_to_bgr_lmdb("train_opt_flow_lmdb")



