import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys, requests, shutil, os
from urllib import request, error
from skimage import io
from skimage.transform import resize
import time

get_ipython().run_line_magic('matplotlib', 'inline')

train = pd.read_csv('./data/train.csv')
print('Train:\t\t', train.shape)

def fetch_image(url):
    """ Get image from given url """
    response=requests.get(url, stream=True)
    
    with open('./data/image_2.jpg', 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
        
    del response

# Download images to ./train/
urls = train['url'].values
t0 = time.time()

# Loop through urls to download images
for idx in range(200000, 300000):
    url = urls[idx]
    # Check if already downloaded
    if os.path.exists('./data/tmp2/' + str(idx) + '.jpg'):
        continue
        
    # Get image from url
    fetch_image(url)
    os.rename('./data/image_2.jpg', './data/tmp2/'+ str(idx) + '.jpg')
    
    # Helpful information
    if idx % 10000 == 0:
        t = time.time() - t0
        print('\nProcess: {:9d}'.format(idx), '   Used time: {} s'.format(np.round(t, 0)))
        t0 = time.time()
    if idx % 125 == 0:
        print('=', end='')



