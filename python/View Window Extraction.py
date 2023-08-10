import sys

sys.path.insert(0, '../src/')

import time
import os
import glob
import numpy as np
import pickle
import xml.etree.ElementTree
import cv2

import window_extraction
import preprocessing
import config
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

e = xml.etree.ElementTree.parse(config.XML_PATH).getroot()
dic = window_extraction.parse_xml(e)
path = '../data/SceneTrialTrain/apanar_06.08.2002/Pict0031.jpg'

srcBGR = cv2.imread(path)
destRGB = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)

plt.imshow(destRGB)

plt.yticks([])
plt.xticks([])
plt.savefig('window_img.png', transparent=True)
plt.show()

window_extraction.extract_random_windows(path, 32, (32, 32), 10, dic, True, True)

window_extraction.extract_random_windows(path, 32, (32, 32), 10, dic, False, True)



