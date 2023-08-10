import requests
import json

# HOST = 'https://api.meerkat.com.br/frapi/'
HOST = 'http://localhost:4444'

p = {}
h = {}

p['accessKey'] = 'b45696c569789cdc57fbe96bee897ce9'
p['email'] = 'contato@meerkat.com.br'
p['subscription'] = 'pro'
h = {'Content-type': 'application/json', 'Accept': 'text/plain'}
res = requests.post(HOST+'/user/create', data=json.dumps(p), headers=h)
print(res)
ores = res.json()
print(ores)
api_key = ores['api_key']

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
from urllib.request import urlopen
import cv2

def plot_detection_info(image, ores):
    for rec in ores['people']:
        l = rec['top_left']['x']
        t = rec['top_left']['y']
        r = rec['bottom_right']['x']
        b = rec['bottom_right']['y']
        tl = [l, t]
        br = [r, b]
        cv2.rectangle(image, (int(tl[0]), int(tl[1])), (int(br[0]), int(br[1])), (255, 0, 0), 5)

    return image

def plot_train_info(image, ores):
    rec = ores['selectedFace']
    l = rec['top_left']['x']
    t = rec['top_left']['y']
    r = rec['bottom_right']['x']
    b = rec['bottom_right']['y']
    tl = [l, t]
    br = [r, b]
    cv2.rectangle(image, (int(tl[0]), int(tl[1])), (int(br[0]), int(br[1])), (255, 0, 0), 5)

    return image

import requests
import json
import os
import cv2
from requests_toolbelt import MultipartEncoder
from pprint import pprint as pp

train_path = './frames'
pp(os.listdir(train_path))


SHOW_DETECTIONS = True
p = {}
par = {}
for label_name in os.listdir(train_path):
    for image_name in os.listdir(train_path+'/'+label_name):
        filename = train_path+'/'+label_name+'/'+image_name
#         if filename[filename.rindex("."):] == ".jpg":
        pp(filename)
        m = MultipartEncoder(
                fields={'image': ('filename', open(filename, 'rb')), 'label': label_name}
        )
        p['label'] = label_name
        res = requests.post(HOST+'/train/person', data=m, headers={'Content-Type': m.content_type, 'api_key': api_key})
        print('res',res)
        ores = res.json()
        if 'error' in ores:
            print(ores)
            continue
        print(res)
        if SHOW_DETECTIONS:
            img = cv2.imread(train_path+'/'+label_name+'/'+image_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = plot_train_info(img, ores)
            plt.imshow(img)
            plt.show()

print(api_key)



