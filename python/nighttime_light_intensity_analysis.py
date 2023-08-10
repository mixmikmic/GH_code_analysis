get_ipython().magic('load_ext autotime')

import os
import glob
import subprocess
import numpy as np
import pandas as pd
import datetime as dt
from pandas.tseries.offsets import *
import seaborn as sns
from osgeo import gdal
from IPython.display import display

import matplotlib.cm as cm
import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')

#plot image function
def plot_images(images,shape=None):
    if shape is None:
        shape = (1, len(images))
    for i, image in enumerate(images):
        s = plt.subplot(shape[0],shape[1],i + 1)
        s.set_xticklabels([])
        s.set_yticklabels([])
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')
        #if image.dim
        if image.ndim == 3:
            plt.imshow(image[:,:,::-1])
        else:
            plt.imshow(image, cmap='gray')
    plt.show()

#cutting geotiff image function
def make_slice(ulx, uly, lrx, lry, in_path, out_path):
    subprocess.call(['gdal_translate', "-projwin", str(ulx), str(uly), str(lrx), str(lry), in_path , out_path+".tif"])

#read geotif and return uint8 image
def read_img(path):
    ds = gdal.Open(path)
    img = np.array(ds.GetRasterBand(1).ReadAsArray())
    #print(img.dtype)
    return img

in_path = "/Users/hayashiyus/Documents/R/imagery_key/"

rasters = glob.glob(in_path + '*.tif')

#  kanto_region
ulx = 138
uly = 37
lrx = 141
lry = 34

tstr = '2012-04'
tdatetime = dt.datetime.strptime(tstr, '%Y-%m')

i = 0

for raster in rasters :
    tdatetime_latest = tdatetime + MonthBegin(i)
    tstr = tdatetime_latest.strftime('%Y-%m')
    out_path ="/Users/hayashiyus/Documents/R/imagery_key/kanto_region/" + "kanto_region_" + tstr # output raster
    print("kanto_region_" + tstr +'.tif')
    make_slice(ulx, uly, lrx, lry, raster, out_path)
    i = i + 1

in_path = "/Users/hayashiyus/Documents/R/imagery_key/"

rasters = glob.glob(in_path + '*.tif')

#  japan islands
ulx = 125
uly = 47
lrx = 153
lry = 24

tstr = '2012-04'
tdatetime = dt.datetime.strptime(tstr, '%Y-%m')

i = 0

for raster in rasters :
    tdatetime_latest = tdatetime + MonthBegin(i)
    tstr = tdatetime_latest.strftime('%Y-%m')
    out_path ="/Users/hayashiyus/Documents/R/imagery_key/japan_islands/" + "japan_islands_" + tstr # output raster
    print("japan_islands_" + tstr +'.tif')
    make_slice(ulx, uly, lrx, lry, raster, out_path)
    i = i + 1

#download VIIRS DNB from https://www.ngdc.noaa.gov/eog/viirs/download_dnb_composites.html
all_path2016 = "/Users/hayashiyus/Documents/R/imagery_key/SVDNB_npp_201604_75N060E_vcmslcfg_v10.avg_rade9.tif"

#read geotiff image
ds = gdal.Open(all_path2016)
#extract light intensity data
img = np.array(ds.GetRasterBand(1).ReadAsArray())
#formatting light intensity data
arr = img.reshape((1,img.size))

print(img.shape)
print(img.dtype)
print(img.size)

with plt.style.context(('ggplot')):
    sns.kdeplot(arr[0],shade=True)

with plt.style.context(('ggplot')):
    sns.kdeplot(np.log(arr[0]),shade=True)

#download VIIRS DNB from https://www.ngdc.noaa.gov/eog/viirs/download_dnb_composites.html
japan_path2016 = "/Users/hayashiyus/Documents/R/imagery_key/japan_islands/japan_islands_2016-04.tif"

#read geotiff image
ds = gdal.Open(japan_path2016)
#extract light intensity data
img = np.array(ds.GetRasterBand(1).ReadAsArray())
#formatting light intensity data
arr = img.reshape((1,img.size))

print(img.shape)
print(img.dtype)
print(img.size)

with plt.style.context(('ggplot')):
    sns.kdeplot(arr[0],shade=True)

with plt.style.context(('ggplot')):
    sns.kdeplot(np.log(arr[0]),shade=True)

bins = np.logspace(0.01, 3, 500)

with plt.style.context(('ggplot')):
    sns.distplot(arr[0], bins=bins, kde=True)
    plt.gca().set_xscale("log")
plt.show()

bins = np.logspace(0.01, 3, 500)

fig = plt.figure()

with plt.style.context(('ggplot')):

    ax1 = fig.add_subplot(111)
    n, bins, patches = ax1.hist(arr[0], bins=bins, color='royalblue') # histogram
    fig.gca().set_xscale("log")
    ax1.set_xlim(arr[0].min(), arr[0].max())

    ax2 = ax1.twinx() # another y-axis
    n, bins, patches = ax2.hist(arr[0], bins=bins, cumulative=True, normed=True, histtype='step', ec='k') # cumulative
    ax2.set_xlim(arr[0].min(), arr[0].max())
    ax2.set_ylim(0, 1)

plt.show()

bins = np.logspace(0.01, 3, 500)

fig = plt.figure()

with plt.style.context(('ggplot')):

    ax1 = fig.add_subplot(111)
    n, bins, patches = ax1.hist(arr[0], bins=bins, log=True, color='royalblue') # histogram
    fig.gca().set_xscale("log")
    ax1.set_xlim(arr[0].min(), arr[0].max())

    ax2 = ax1.twinx() # another y-axis
    n, bins, patches = ax2.hist(np.log(arr[0]), bins=bins, cumulative=True, normed=True, histtype='step', ec='k') # cumulative
    ax2.set_xlim(arr[0].min(), arr[0].max())
    ax2.set_ylim(0, 1)

plt.show()

bins = np.logspace(0.01, 3, 500)

fig = plt.figure()

with plt.style.context(('ggplot')):

    ax1 = fig.add_subplot(111)
    n, bins, patches = ax1.hist(arr[0], bins=bins, log=True, color='royalblue') # histogram
    fig.gca().set_xscale("log")
    ax1.set_xlim(arr[0].min(), arr[0].max())

    ax2 = ax1.twinx() # another y-axis
    n, bins, patches = ax2.hist(np.log(arr[0]), bins=bins, normed=True, histtype='step', ec='k') # cumulative
    ax2.set_xlim(arr[0].min(), arr[0].max())
    ax2.set_ylim(0, 1)

plt.show()

path2016 = "/Users/hayashiyus/Documents/R/imagery_key/japan_islands/japan_islands_2016-04.tif"
img2016 = read_img(path2016)
path2015 = "/Users/hayashiyus/Documents/R/imagery_key/japan_islands/japan_islands_2015-04.tif"
img2015 = read_img(path2015)
path2014 = "/Users/hayashiyus/Documents/R/imagery_key/japan_islands/japan_islands_2014-04.tif"
img2014 = read_img(path2014)
path2013 = "/Users/hayashiyus/Documents/R/imagery_key/japan_islands/japan_islands_2013-04.tif"
img2013 = read_img(path2013)
path2012 = "/Users/hayashiyus/Documents/R/imagery_key/japan_islands/japan_islands_2012-04.tif"
img2012 = read_img(path2012)

path2016_kanto = "/Users/hayashiyus/Documents/R/imagery_key/kanto_region/kanto_region_2016-04.tif"
img2016_kanto = read_img(path2016_kanto)
path2015_kanto = "/Users/hayashiyus/Documents/R/imagery_key/kanto_region/kanto_region_2015-04.tif"
img2015_kanto = read_img(path2015_kanto)

#making light intensity data
img1615 = img2016 - img2015
img1614 = img2016 - img2014
img1613 = img2016 - img2013
img1612 = img2016 - img2012

img1514 = img2015 - img2014
img1413 = img2014 - img2013
img1312 = img2013 - img2012

#formatting light intensity data
arr_diff1615 = img1615.reshape((1,img1615.size))
arr_diff1614 = img1614.reshape((1,img1614.size))
arr_diff1613 = img1613.reshape((1,img1613.size))
arr_diff1612 = img1612.reshape((1,img1612.size))

arr_diff1514 = img1514.reshape((1,img1514.size))
arr_diff1413 = img1413.reshape((1,img1413.size))
arr_diff1312 = img1312.reshape((1,img1312.size))

plt.figure(figsize = (15,10))

with plt.style.context(('ggplot')):
    plt.hist(arr_diff1615[0], bins=2000, log = True, alpha=0.3)
    plt.hist(arr_diff1614[0], bins=2000, log = True, alpha=0.3)
    plt.hist(arr_diff1613[0], bins=2000, log = True, alpha=0.3)
    plt.hist(arr_diff1612[0], bins=2000, log = True, alpha=0.3)
    
#plt.xscale('log')
plt.xlim(-500,500)
plt.ylim(0,30000)
plt.show()

plt.figure(figsize = (15,10))

with plt.style.context(('ggplot')):
    plt.hist(arr_diff1615[0], bins=2000, log = True, alpha=0.3)
    plt.hist(arr_diff1514[0], bins=2000, log = True, alpha=0.3)
    plt.hist(arr_diff1413[0], bins=2000, log = True, alpha=0.3)
    plt.hist(arr_diff1612[0], bins=2000, log = True, alpha=0.3)
    
#plt.xscale('log')
plt.xlim(-500,500)
plt.ylim(0,30000)
plt.show()

img1615_kanto = img2016_kanto - img2015_kanto

plt.figure(figsize = (15,10))
plt.imshow(img1615_kanto, clim=(-200, 200), cmap = 'seismic')
plt.colorbar()
plt.show()

img2016_kanto[img2016_kanto < 0] = 0

plt.figure(figsize = (15,10))
plt.imshow(img2016_kanto, clim=(0, 100), cmap = 'hot')
plt.colorbar()
plt.show()

img1615_kanto = img2016_kanto - img2015_kanto
img1615_kanto[img1615_kanto < 0] = 0

plt.figure(figsize = (15,10))
plt.imshow(img1615_kanto, clim=(0, 100), cmap = 'hot')
plt.colorbar()
plt.show()

plt.figure(figsize = (15,10))
plt.imshow(img2016_kanto, clim=(0, 400), cmap = 'hot')
plt.colorbar()
plt.show()

plt.figure(figsize = (15,10))
plt.imshow(img2016_kanto, clim=(0, 400), cmap = 'hot')
plt.xlim(400,500)
plt.ylim(400,300)
plt.colorbar()
plt.show()

img2016_kanto[img2016_kanto < 20] = 0

plt.figure(figsize = (15,10))
plt.imshow(img2016_kanto, clim=(0, 400), cmap = 'hot')
plt.colorbar()
plt.show()

img2016_kanto[img2016_kanto < 150] = 0

plt.figure(figsize = (15,10))
plt.imshow(img2016_kanto, clim=(0, 400), cmap = 'hot')
plt.colorbar()
plt.show()

plt.figure(figsize = (15,10))
plt.imshow(img2016_kanto, clim=(0, 400), cmap = 'hot')
plt.xlim(400,500)
plt.ylim(400,300)
plt.colorbar()
plt.show()

'''
def crop(img, x1, x2, y1, y2):
    """
    Return the cropped image at the x1, x2, y1, y2 coordinates
    """
    if x2 == -1:
        x2=img.shape[1]-1
    if y2 == -1:
        y2=img.shape[0]-1

    mask = np.zeros(img.shape)
    mask[y1:y2+1, x1:x2+1]=1
    m = mask>0

    return img[m].reshape((y2+1-y1, x2+1-x1))

img_cropped = crop(img, 240, 290, 255, 272)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.imshow(img)
ax2.imshow(img_cropped)

plt.show()
'''

plt.figure(figsize = (10,10))

plt.subplot(111)
plt.imshow(img2016, clim=(0, 10), cmap='Purples')
plt.colorbar()
plt.show()

path2016_kanto = "/Users/hayashiyus/Documents/R/imagery_key/kanto_region/kanto_region_2016-04.tif"
img2016_kanto = read_img(path2016_kanto)

plt.figure(figsize = (10,10))

plt.subplot(111)
plt.imshow(img2016_kanto, clim=(0, 75), cmap='Purples')
plt.colorbar()
plt.show()

plt.figure(figsize = (10,10))

plt.subplot(111)
plt.imshow(img1612, clim=(-5, 5), cmap='seismic')
plt.colorbar()
plt.show()

import fiona
import rasterio
import rasterio.mask
import pandas as pd
from itertools import product

def mask_image(vector_file, raster_file, output_name):
   
    features = [vector_file]

    #masking
    with rasterio.open(raster_file) as src:
        out_image, out_transform = rasterio.mask.mask(src, features,
                                                            crop=True,nodata=0)
        out_meta = src.meta.copy()

    #write out masked image
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})
    with rasterio.open(output_name, "w", **out_meta) as dest:
        dest.write(out_image)
    return out_image

#shp file from https://www.esrij.com/products/japan-shp/
vector = '/Users/hayashiyus/Documents/R/imagery_key/japan_ver80/japan_ver80.shp'
v = fiona.open(vector)
#otaku is number 673
print(v[673]['properties'][u'CITY_ENG'])

'''
%%time
outputdata = []
outputindex = []
outputdir  = '/Users/hayashiyus/Documents/R/imagery_key/japan_islands/number_673/'

year = ['2012']
month = ['04','05','06','07','08','09','10','11','12']
#please change this part accordingly to iterate over daily images
for y,m in product(year,month):

    rasterdir = '/Users/hayashiyus/Documents/R/imagery_key/japan_islands/small_16000-6800-5500-4200_VIIRS-%s-%s.tif' %(y,m)
    number = '%s-%s'%(m,y)
    src = rasterio.open(rasterdir)
    out_image = mask_image(v[673]['geometry'], rasterdir, outputdir+"shibuyaku_VIIRS-%s-%s.tif" %number)
    data = out_image.data.reshape(-1)
    #drop parts that have no data
    data = data[~np.isnan(data)]
    outputdata.append(data)
    outputindex.append(number)
    
output = pd.DataFrame(data = outputdata, index = outputindex)
output.to_csv(outputdir+'shibuyaku.csv')
'''

'''
%%time
outputdata = []
outputindex = []
outputdir  = '/Users/hayashiyus/Documents/R/imagery_key/japan_islands/number_673/'

year = ['2013','2014', '2015','2016']
month = ['01','02','03','04','05','06','07','08','09','10','11','12']
#please change this part accordingly to iterate over daily images
for y,m in product(year,month):

    rasterdir = '/Users/hayashiyus/Documents/R/imagery_key/japan_islands/small_16000-6800-5500-4200_VIIRS-%s-%s.tif' %(y,m)
    number = '%s-%s'%(m,y)
    src = rasterio.open(rasterdir)
    out_image = mask_image(v[673]['geometry'], rasterdir, outputdir+"shibuyaku_VIIRS-%s-%s.tif" %number)
    data = out_image.data.reshape(-1)
    #drop parts that have no data
    data = data[~np.isnan(data)]
    outputdata.append(data)
    outputindex.append(number)
    
output = pd.DataFrame(data = outputdata, index = outputindex)
output.to_csv(outputdir+'shibuyaku.csv')
'''

'''
%%time
outputdata = []
outputindex = []
outputdir  = '/Users/hayashiyus/Documents/R/imagery_key/japan_islands/number_673/'

year = ['2017']
month = ['01','02','03','04']
#please change this part accordingly to iterate over daily images
for y,m in product(year,month):

    rasterdir = '/Users/hayashiyus/Documents/R/imagery_key/japan_islands/small_16000-6800-5500-4200_VIIRS-%s-%s.tif' %(y,m)
    number = '%s-%s'%(m,y)
    src = rasterio.open(rasterdir)
    out_image = mask_image(v[673]['geometry'], rasterdir, outputdir+"shibuyaku_VIIRS-%s-%s.tif" %number)
    data = out_image.data.reshape(-1)
    #drop parts that have no data
    data = data[~np.isnan(data)]
    outputdata.append(data)
    outputindex.append(number)
    
output = pd.DataFrame(data = outputdata, index = outputindex)
output.to_csv(outputdir+'shibuyaku.csv')
'''

#shp file from https://www.esrij.com/products/japan-shp/
vector = '/Users/hayashiyus/Documents/R/imagery_key/japan_ver80/japan_ver80.shp'
v = fiona.open(vector)
#otaku is number 671
print(v[671]['properties'][u'CITY_ENG'])

outputdata = []
outputindex = []
outputdir  = '/Users/hayashiyus/Documents/R/imagery_key/DMSP_imagery/number_671/'

satellite_model = ['10']
year= ['1992','1993','1994']
#please change this part accordingly to iterate over daily images
for m,y in product(satellite_model,year):

    rasterdir = '/Users/hayashiyus/Documents/R/imagery_key/DMSP_imagery/F%s%s.tif' %(m,y)
    number = '%s-%s'%(m,y)
    src = rasterio.open(rasterdir)
    out_image = mask_image(v[671]['geometry'], rasterdir, outputdir+"otaku_F%s.tif" %number)
    data = out_image.data.reshape(-1)
    #drop parts that have no data
    data = data[~np.isnan(data)]
    outputdata.append(data)
    outputindex.append(number)
    
output = pd.DataFrame(data = outputdata, index = outputindex)
output.to_csv(outputdir+'otaku.csv')

outputdata = []
outputindex = []
outputdir  = '/Users/hayashiyus/Documents/R/imagery_key/DMSP_imagery/number_671/'

satellite_model = ['12']
year= ['1995','1996','1997','1998','1999']
#please change this part accordingly to iterate over daily images
for m,y in product(satellite_model,year):

    rasterdir = '/Users/hayashiyus/Documents/R/imagery_key/DMSP_imagery/F%s%s.tif' %(m,y)
    number = '%s-%s'%(m,y)
    src = rasterio.open(rasterdir)
    out_image = mask_image(v[671]['geometry'], rasterdir, outputdir+"otaku_F%s.tif" %number)
    data = out_image.data.reshape(-1)
    #drop parts that have no data
    data = data[~np.isnan(data)]
    outputdata.append(data)
    outputindex.append(number)
    
output = pd.DataFrame(data = outputdata, index = outputindex)
output.to_csv(outputdir+'otaku.csv')

outputdata = []
outputindex = []
outputdir  = '/Users/hayashiyus/Documents/R/imagery_key/DMSP_imagery/number_671/'

satellite_model = ['15']
year= ['2000','2001','2002','2003','2004','2005','2006','2007']
#please change this part accordingly to iterate over daily images
for m,y in product(satellite_model,year):

    rasterdir = '/Users/hayashiyus/Documents/R/imagery_key/DMSP_imagery/F%s%s.tif' %(m,y)
    number = '%s-%s'%(m,y)
    src = rasterio.open(rasterdir)
    out_image = mask_image(v[671]['geometry'], rasterdir, outputdir+"otaku_F%s.tif" %number)
    data = out_image.data.reshape(-1)
    #drop parts that have no data
    data = data[~np.isnan(data)]
    outputdata.append(data)
    outputindex.append(number)
    
output = pd.DataFrame(data = outputdata, index = outputindex)
output.to_csv(outputdir+'otaku.csv')

outputdata = []
outputindex = []
outputdir  = '/Users/hayashiyus/Documents/R/imagery_key/DMSP_imagery/number_671/'

satellite_model = ['16']
year= ['2004','2005','2006','2007','2008','2009']
#please change this part accordingly to iterate over daily images
for m,y in product(satellite_model,year):

    rasterdir = '/Users/hayashiyus/Documents/R/imagery_key/DMSP_imagery/F%s%s.tif' %(m,y)
    number = '%s-%s'%(m,y)
    src = rasterio.open(rasterdir)
    out_image = mask_image(v[671]['geometry'], rasterdir, outputdir+"otaku_F%s.tif" %number)
    data = out_image.data.reshape(-1)
    #drop parts that have no data
    data = data[~np.isnan(data)]
    outputdata.append(data)
    outputindex.append(number)
    
output = pd.DataFrame(data = outputdata, index = outputindex)
output.to_csv(outputdir+'otaku.csv')

outputdata = []
outputindex = []
outputdir  = '/Users/hayashiyus/Documents/R/imagery_key/DMSP_imagery/number_671/'

satellite_model = ['18']
year= ['2010','2011','2012','2013']
#please change this part accordingly to iterate over daily images
for m,y in product(satellite_model,year):

    rasterdir = '/Users/hayashiyus/Documents/R/imagery_key/DMSP_imagery/F%s%s.tif' %(m,y)
    number = '%s-%s'%(m,y)
    src = rasterio.open(rasterdir)
    out_image = mask_image(v[671]['geometry'], rasterdir, outputdir+"otaku_F%s.tif" %number)
    data = out_image.data.reshape(-1)
    #drop parts that have no data
    data = data[~np.isnan(data)]
    outputdata.append(data)
    outputindex.append(number)
    
output = pd.DataFrame(data = outputdata, index = outputindex)
output.to_csv(outputdir+'otaku.csv')

display(output)

path = "/Users/hayashiyus/Documents/R/imagery_key/"

rasters = glob.glob(path + '*.tif')
radiance = []
fnames = []
for raster in rasters :
    fname = raster.split("/")[-1]
    print(fname)
    fnames.append(fname)
    ds = gdal.Open(path + fname)
    img = np.array(ds.GetRasterBand(1).ReadAsArray())
    quantile = np.percentile(img, [90, 95, 99, 50])
    img = img[img>=quantile[1]]
    radiance.append(np.sum(img))

# radiance_range
radiance_range = int(len(radiance)/3)

sum_radiance = [np.sum(radiance[0+(3*x):3+(3*x)])/3  for x in range(radiance_range)]

print('raw data')
display(sum_radiance)

sum_radiance [0] = np.sum(radiance[0:2])/2
sum_radiance [1] = np.sum(radiance[4:6])/2
sum_radiance [4] = np.sum(radiance[12:14])/2
sum_radiance [5] = np.sum(radiance[16:18])/2

print('processed data')
display(sum_radiance)

path = "/Users/hayashiyus/Documents/R/imagery_key/japan_islands/"

rasters = glob.glob(path + '*.tif')
radiance = []
fnames = []
for raster in rasters :
    fname = raster.split("/")[-1]
    print(fname)
    fnames.append(fname)
    ds = gdal.Open(path + fname)
    img = np.array(ds.GetRasterBand(1).ReadAsArray())
    quantile = np.percentile(img, [90, 95, 99, 50])
    img = img[img>=quantile[1]]
    radiance.append(np.sum(img))

# radiance_range
radiance_range = int(len(radiance)/3)

sum_radiance = [np.sum(radiance[0+(3*x):3+(3*x)])/3  for x in range(radiance_range)]

print('raw data')
display(sum_radiance)

sum_radiance [0] = np.sum(radiance[0:2])/2
sum_radiance [1] = np.sum(radiance[4:6])/2
sum_radiance [4] = np.sum(radiance[12:14])/2
sum_radiance [5] = np.sum(radiance[16:18])/2

print('processed data')
display(sum_radiance)



