# Testing the environment

import cv2
import numpy as np
import tensorflow as tf

print("OpenCV version", cv2.__version__)

print("Numpy version", np.__version__)

print("Tensorflow version", tf.__version__)

import requests
import json
import cv2
import ast
import requests
import urllib.parse
import numpy as np
from matplotlib import pyplot as plt

get_ipython().magic('matplotlib inline')


# Google Client API Key
API_KEY = "<your client key>"
IS_AVAILABLE = True

GMAPS_URL = 'https://maps.googleapis.com/maps/api/staticmap?'

SUCCESS_STATUS = 'success'
lat  =  '-47.7538' 
long =  '-91.4351'

LAT_LONG = "http://api.open-notify.org/iss-now.json"

r = requests.get(LAT_LONG)

location = json.loads(r.text)

print(type(location))

if 'message' in location:
    if location['message'] == SUCCESS_STATUS:
        lat  = location['iss_position']['latitude']
        long = location['iss_position']['longitude']        
else:
    print("Try to run again")
    IS_AVAILABLE = False

if IS_AVAILABLE:
    # do the fetching of tile
    print(lat,long)
   
    mydict = {
               'size': '600x400',
               'zoom':'12',
               'center':  ",".join([lat,long]),
               'maptype': 'satellite',
               'key': API_KEY
              }

    url = GMAPS_URL + urllib.parse.urlencode(mydict)
    
    
    print(url)
    
    req = requests.get(url)
    arr = np.asarray(bytearray(req.content), dtype=np.uint8)
    img = cv2.imdecode(arr,-1) # 'load it as it is'
    print(img.shape)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    fig = plt.figure("Demo",(10,10))
    plt.imshow(img)
    plt.show()

import threading
import os
import binascii
import json
import requests

SHOTS = 50
SEED = "001"
SAVE_SRC = os.path.join("..","images","Aerial_Panorama","001")
LAT_LONG = "http://api.open-notify.org/iss-now.json"

LOG_FILE = os.path.join(SAVE_SRC,"data_log.log")

# Google Client API Key
API_KEY = "<your client key>"

GMAPS_URL = 'https://maps.googleapis.com/maps/api/staticmap?'

log_f = open(LOG_FILE,'w')

xcount = 0

def save_new_frame(count):
 
    IS_AVAILABLE = True
    
    r = requests.get(LAT_LONG)

    location = json.loads(r.text)

    if 'message' in location:
        if location['message'] == SUCCESS_STATUS:
            lat  = location['iss_position']['latitude']
            long = location['iss_position']['longitude']        
    else:
        print("Try to run again")
        IS_AVAILABLE = False

    if IS_AVAILABLE:
        # do the fetching of tile
        mydict = {
                   'size': '600x400',
                   'zoom':'8',
                   'center':  ",".join([lat,long]),
                   'maptype': 'satellite',
                   'key': API_KEY
                  }

        url = GMAPS_URL + urllib.parse.urlencode(mydict)

        req = requests.get(url)
        arr = np.asarray(bytearray(req.content), dtype=np.uint8)
        img = cv2.imdecode(arr,-1) # 'load it as it is'
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        name = str(binascii.b2a_hex(os.urandom(15)))
        fname = os.path.join(SAVE_SRC, name + ".jpg")
        
        cv2.imwrite(fname,img)
        log_f.write("/t".join([str(i) for i in (lat,long,name)]) + "\n")
    

def printit():
  count = 0
  threading.Timer(1.0, printit).start()
  count+=1
  save_new_frame(count)
  
  if count == SHOTS:
     return

printit()
#

from threading import Timer

class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer     = None
        self.interval   = interval
        self.function   = function
        self.args       = args
        self.kwargs     = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False

import threading
import os
import binascii
import requests
import json
import cv2
import ast
import requests
import urllib.parse
import numpy as np
from matplotlib import pyplot as plt

get_ipython().magic('matplotlib inline')

# Google Client API Key
API_KEY = "<your client code>"
IS_AVAILABLE = True

GMAPS_URL = 'https://maps.googleapis.com/maps/api/staticmap?'

SUCCESS_STATUS = 'success'

SHOTS = 50
SEED = "004"
SAVE_SRC = os.path.join("..","images","Aerial_Panorama","004")
LAT_LONG = "http://api.open-notify.org/iss-now.json"

LOG_FILE = os.path.join(SAVE_SRC,"data_log.log")

# Google Client API Key
API_KEY = "AIzaSyCgr7etuv6TgndjC8UV5-DG6nwKXeqwk9k"

GMAPS_URL = 'https://maps.googleapis.com/maps/api/staticmap?'

log_f = open(LOG_FILE,'w')

xcount = 0

def save_new_frame():
 
    IS_AVAILABLE = True
    
    r = requests.get(LAT_LONG)

    location = json.loads(r.text)

    if 'message' in location:
        if location['message'] == SUCCESS_STATUS:
            lat  = location['iss_position']['latitude']
            long = location['iss_position']['longitude']        
    else:
        print("Try to run again")
        IS_AVAILABLE = False

    if IS_AVAILABLE:
        # do the fetching of tile
        mydict = {
                   'size': '600x400',
                   'zoom':'12',
                   'center':  ",".join([lat,long]),
                   'maptype': 'satellite',
                   'key': API_KEY
                  }

        url = GMAPS_URL + urllib.parse.urlencode(mydict)

        req = requests.get(url)
        arr = np.asarray(bytearray(req.content), dtype=np.uint8)
        img = cv2.imdecode(arr,-1) # 'load it as it is'
        
        name = str(binascii.b2a_hex(os.urandom(15)).decode('ascii'))
        fname = os.path.join(SAVE_SRC, name + ".jpg")
        
        cv2.imwrite(fname,img)
        log_f.write("\t".join([str(i) for i in (lat,long,name)]) + "\n")
    

from time import sleep

print("starting...")
rt = RepeatedTimer(1, save_new_frame) # it auto-starts, no need of rt.start()
try:
    sleep(SHOTS) # your long-running job goes here...
finally:
    log_f.close()
    rt.stop() # better in a try/finally block to make sure the program ends!

import os

SAVE_SRC = os.path.join("..","images","Aerial_Panorama","004")

LOG_FILE = os.path.join(SAVE_SRC,"data_log.log")

lat_lst = []
long_lst = []

# saving the lat,long as a list
with open(LOG_FILE) as fs:
    for line in fs:
        lat,long,img_id = line.split("\t")
        
        lat = float(lat)
        long = float(long)
        
        #ll_lst.append((lat,long))
        lat_lst.append(lat)
        long_lst.append(long)

plt.title("Visualising the Path of International Space Station")       
plt.scatter(lat_lst,long_lst)

plt.xlabel("Latitude")
plt.ylabel("Longitude")

import cv2
import numpy as np 

class matchers:
	def __init__(self):
		self.surf = cv2.xfeatures2d.SURF_create()
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm=0, trees=5)
		search_params = dict(checks=50)
		self.flann = cv2.FlannBasedMatcher(index_params, search_params)

	def match(self, i1, i2, direction=None):
		imageSet1 = self.getSURFFeatures(i1)
		imageSet2 = self.getSURFFeatures(i2)
		print("Direction : ", direction)
		matches = self.flann.knnMatch(
			imageSet2['des'],
			imageSet1['des'],
			k=2
			)
		good = []
		for i , (m, n) in enumerate(matches):
			if m.distance < 0.7*n.distance:
				good.append((m.trainIdx, m.queryIdx))

		if len(good) > 4:
			pointsCurrent = imageSet2['kp']
			pointsPrevious = imageSet1['kp']

			matchedPointsCurrent = np.float32(
				[pointsCurrent[i].pt for (__, i) in good]
			)
			matchedPointsPrev = np.float32(
				[pointsPrevious[i].pt for (i, __) in good]
				)

			H, s = cv2.findHomography(matchedPointsCurrent, matchedPointsPrev, cv2.RANSAC, 4)
			return H
		return None

	def getSURFFeatures(self, im):
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		kp, des = self.surf.detectAndCompute(gray, None)
		return {'kp':kp, 'des':des}

import os
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

class Stitch:
	def __init__(self, args):
		self.path = args
		fp = open(self.path, 'r')
		
		filenames = [ os.path.join("..","images","Aerial_Panorama","005",line.split()[2] + ".jpg") for line in  fp.readlines()][:6]
		print(filenames)
		self.images = [cv2.resize(cv2.imread(each),(480, 320)) for each in filenames]
		print([x.shape for x in self.images])
		self.count = len(self.images)
		self.left_list, self.right_list, self.center_im = [], [],None
		self.matcher_obj = matchers()
		self.prepare_lists()

	def prepare_lists(self):
		
		print("Number of images : %d"%self.count)
		self.centerIdx = int(self.count/2) 
		
		print("Center index image : {0}".format(self.centerIdx))
		self.center_im = self.images[self.centerIdx]
		for i in range(self.count):
			if(i<=self.centerIdx):
				self.left_list.append(self.images[i])
			else:
				self.right_list.append(self.images[i])
		print("Image lists prepared")

	def leftshift(self):
		a = self.left_list[0]
		for b in self.left_list[1:]:
			H = self.matcher_obj.match(a, b, 'left')
			print("Homography is : ", H)
			xh = np.linalg.inv(H)
			print("Inverse Homography :", xh)
			ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]));
			ds = ds/ds[-1]
			print("final ds=>", ds)
			f1 = np.dot(xh, np.array([0,0,1]))
			f1 = f1/f1[-1]
			xh[0][-1] += abs(f1[0])
			xh[1][-1] += abs(f1[1])
			ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
			offsety = abs(int(f1[1]))
			offsetx = abs(int(f1[0]))
			dsize = (int(ds[0])+offsetx, int(ds[1]) + offsety)
			print("image dsize =>", dsize)
			tmp = cv2.warpPerspective(a, xh, dsize)
			tmp[offsety:b.shape[0]+offsety, offsetx:b.shape[1]+offsetx] = b
			a = tmp

		self.leftImage = tmp

		
	def rightshift(self):
		for each in self.right_list:
			H = self.matcher_obj.match(self.leftImage, each, 'right')
			print("Homography :", H)
			txyz = np.dot(H, np.array([each.shape[1], each.shape[0], 1]))
			txyz = txyz/txyz[-1]
			dsize = (int(txyz[0])+self.leftImage.shape[1], int(txyz[1])+self.leftImage.shape[0])
			tmp = cv2.warpPerspective(each, H, dsize)
			tmp = self.mix_and_match(self.leftImage, tmp)
			print("tmp shape",tmp.shape)
			print("self.leftimage shape=", self.leftImage.shape)
			self.leftImage = tmp
		


	def mix_and_match(self, leftImage, warpedImage):
		i1y, i1x = leftImage.shape[:2]
		i2y, i2x = warpedImage.shape[:2]
		print(leftImage[-1,-1])

		t = time.time()
		black_l = np.where(leftImage == np.array([0,0,0]))
		black_wi = np.where(warpedImage == np.array([0,0,0]))
		print(time.time() - t)
		print(black_l[-1])

		for i in range(0, i1x):
			for j in range(0, i1y):
				try:
					if(np.array_equal(leftImage[j,i],np.array([0,0,0])) and  np.array_equal(warpedImage[j,i],np.array([0,0,0]))):
						# print "BLACK"
						# instead of just putting it with black, 
						# take average of all nearby values and avg it.
						warpedImage[j,i] = [0, 0, 0]
					else:
						if(np.array_equal(warpedImage[j,i],[0,0,0])):
							# print "PIXEL"
							warpedImage[j,i] = leftImage[j,i]
						else:
							if not np.array_equal(leftImage[j,i], [0,0,0]):
								bw, gw, rw = warpedImage[j,i]
								bl,gl,rl = leftImage[j,i]
								# b = (bl+bw)/2
								# g = (gl+gw)/2
								# r = (rl+rw)/2
								warpedImage[j, i] = [bl,gl,rl]
				except:
					pass
		# cv2.imshow("waRPED mix", warpedImage)
		# cv2.waitKey()
		return warpedImage
    
IMG_PATH = os.path.join("..","images","Aerial_Panorama","005","data_log.log")

s = Stitch(IMG_PATH)
s.leftshift()
s.rightshift()
print("done")

fig = plt.figure("Aerial Montage", (15,15))

out_img = s.leftImage

out_img   = out_img[0:800,0:950,:]
plt.imshow(out_img)

plt.show()



