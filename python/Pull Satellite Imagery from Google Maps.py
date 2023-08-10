import sys, os
import urllib2, httplib

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import pandas as pd

from IPython.display import clear_output

def construct_googlemaps_url_request(center=None, zoom=None, imgsize=(500,500),                                      maptype="roadmap", apiKey="", imgformat="jpeg"):
    request =  "http://maps.google.com/maps/api/staticmap?" # base URL, append query params, separated by &
    if center is not None:
        request += "center=%s&"%center.replace(" ","+")
    if zoom is not None:
        request += "zoom=%d&"%zoom  # zoom 0 (all of the world scale ) to 22 (single buildings scale)
    if apiKey is not None:
        request += "key=%s&"%apiKey
    request += "size=%dx%d&"%imgsize  # tuple of ints, up to 640 by 640
    request += "format=%s&"%imgformat
    request += "maptype=%s&sensor=false"%maptype  # roadmap, satellite, hybrid, terrain
    return request

url = construct_googlemaps_url_request(center="Berkeley, CA")
url

import urllib
import cStringIO 
import Image

def get_static_google_map(request, filename=None):  
    if filename is not None:
        urllib.urlretrieve(request, filename) 
    else:
        web_sock = urllib.urlopen(request)
        imgdata = cStringIO.StringIO(web_sock.read()) # constructs a StringIO holding the image
        try:
            img = Image.open(imgdata)
        except IOError:
            print "IOError:", imgdata.read() # print error (or it may return a image showing the error"
            return None
        else:
            return np.asarray(img.convert("RGB"))

url = construct_googlemaps_url_request(center="Berkeley, CA", maptype="satellite", zoom=19)
img = get_static_google_map(url)

plt.figure(figsize=(7,7))
plt.imshow(img)

sites_df = pd.read_csv("/home/ubuntu/data/prop39/site_information.csv",                        converters={'Site CDS Code':str, 'Site ZIP Code':str})
sites_df.head()

locations = dict(zip(sites_df['Site CDS Code'].values,
                 map(lambda x, y: "%2.6f,%2.6f"%(x,y), sites_df['lat'].values, sites_df['lon'].values)))

myAPIKey   = "AIzaSyAmgXPgd-Db8HD_juxtPf_4nricPBdcOrw"
myOutPath  = "/home/ubuntu/data/prop39/images/"
zoomLevels = [17, 18, 19]

url = construct_googlemaps_url_request(center=locations.values()[0], maptype="satellite", zoom=18, apiKey=myAPIKey)
img = get_static_google_map(url)
plt.figure(figsize=(7,7))
plt.imshow(img)

url = construct_googlemaps_url_request(center=locations.values()[0], maptype="satellite", zoom=18, apiKey=myAPIKey)
img = get_static_google_map(url, filename=myOutPath+"/%s.jpg"%locations.keys()[0])

if not os.path.exists(myOutPath):
    os.makedirs(myOutPath)
for zoom in zoomLevels:
    curDir = myOutPath + "/zoom%s/"%zoom
    if not os.path.exists(curDir):
        os.makedirs(curDir)

for i,row in sites_df.iloc[2837:].iterrows():
    clear_output(wait=True)
    siteId   = row['Site CDS Code']
    siteName = row['Site Name']
    if type(siteName) == str:
        siteName = siteName.replace(" ", "").replace(" ", "").replace(".","").replace("/","")
    else:
        siteName = "NONE"
    loc = "%2.6f,%2.6f"%(row['lat'], row['lon'])
    for zoom in zoomLevels:
        filename = filename=myOutPath + "/zoom%s/%s_%s_z%d.jpg"%(zoom, siteId, siteName, zoom)
        url = construct_googlemaps_url_request(center=loc, maptype="satellite", zoom=zoom, apiKey=myAPIKey)
        get_static_google_map(url, filename)
        print "%d/%d : %s (zoom: %d)"%(i, len(sites_df), siteName, zoom)

i

row['Site Name']



