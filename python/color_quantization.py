import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import os
import sys
from PIL import Image
import matplotlib.patches as mpatches
import colorsys
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from sklearn.cluster import MeanShift, estimate_bandwidth

# tgtroot = '/media/senseable-beast/beast-brain-1/Data/streetviewdata/img/'
tgtroot = 'C:/Users/lezhi/Dropbox/thesis/img/'
dataroot = 'C:/Users/lezhi/Dropbox/thesis/data/'

cat_labels = ["Sky", "Building", "Pole", "Unlabelled", "Road",         "Pavement", "Tree", "SignSymbol", "Fence",         "Car", "Pedestrian", "Bicyclist"]

# def getmask(a, **kwargs):
#     inds = [labels.index(c)+243 for c in kwargs['category']]    
#     return [(ele in inds) for ele in np.nditer(a)]

# much better performance than the commented method...:
'''input a 2D index matrix, return a 2D boolean matrix 
where True means the pixel belongs to one of the specified "category".'''

def getmask(a, **kwargs):    
    inds = [cat_labels.index(c)+243 for c in kwargs['category']] 
    # print np.array(inds)-243
    indicators = np.zeros((a.shape[0], a.shape[1], len(inds))).astype(np.uint8)
    for i in range(len(inds)):
        indicators[:,:,i] = np.array(np.squeeze([a==inds[i]]))
    return np.any(indicators, axis=2)

df = pd.DataFrame()

def iter_dir_4(rootdir, dostuff):
    
    citynames = np.array(sorted([d for d in os.listdir(rootdir) if os.path.isdir(rootdir)]))
    for cityname in citynames[np.array([9])]:   ######################
        print cityname
        citypath = rootdir + cityname
        imgnames = sorted([f[:-4] for f in os.listdir(citypath) if os.path.isfile(os.path.join(citypath, f))])
        
        lat_lng_dir = np.array([name.replace('_',',').split(',')[:2] for name in imgnames])
        df1 = pd.DataFrame(lat_lng_dir, columns=['lat', 'lng']).astype(str)############################################
        #df1['city'] = cityname
        df1['imgnames'] = [cityname + "/" + i+".png" for i in imgnames]
        df1gb = df1.groupby(['lat','lng'])
        
        df = df1gb.agg({"imgnames": lambda x: tuple(x)}).reset_index()
    
        records = df['imgnames'].apply(dostuff)
        del df['imgnames']
#         print records.values        
        
        df2 = pd.DataFrame.from_records(list(records.values), columns = ['H', 'S', 'V', 'color'])
#         print df2
    
        df = pd.concat([df, df2], axis=1)
        df.to_csv(dataroot + 'color4_' + cityname +'.csv')
        

# http://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html

# each row is a tuple of image names
def cal_color_4(row):
    imarr = np.zeros([360, 480*len(row), 4]).astype(np.uint8) 
    for i in range(len(row)):
        indi_imarr = np.array(Image.open(tgtroot+ row[i])) # 51.45386,-0.138461068985_
        imarr[:, i*480 : (i+1)*480, :] = indi_imarr

    alpha = imarr[:, :, 3]
    
    mask = getmask(imarr[:,:,3], category=["Building", "SignSymbol", "Pole"]) ##
    if np.sum(mask) < 70000: ###################
        return (np.nan, np.nan, np.nan, np.nan)
    else:
        base = (0*np.ones((360, 480*len(row), 3))).astype(np.uint8)
        base[mask] = (imarr[:,:,:3])[mask]
        color_array = (imarr[:,:,:3])[mask]
        
        def not_black(hsv):         
            return hsv[2] > 0.001
        majority_func = lambda s: s[2]
        ms_quantile = 0.12

        image = (base.astype(np.float64)) /255

        w, h, d = original_shape = tuple(image.shape)
        assert d == 3
        image_array = np.reshape(image, (w * h, d))
        image_array_sample = shuffle(image_array, random_state=0)[:1000]

        bandwidth = estimate_bandwidth(image_array_sample, quantile=ms_quantile)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(image_array_sample)
        labels = ms.predict(image_array)
        labels_unique = np.unique(labels)
        n_clusters = len(labels_unique)
        centers = ms.cluster_centers_

        centers_hsv = [colorsys.rgb_to_hsv(i[0],i[1],i[2]) for i in centers]

        # tuples of index, color, number of pixels with that color
        sum_info = [(i, centers_hsv[i], np.sum(labels == i)) for i in range(len(centers_hsv))]
        non_black = [i for i in sum_info if not_black(i[1])]
        fq_c = max(non_black , key=majority_func)  

        def rgb_to_hex(rgb):
            rgb_ = tuple(((np.array(rgb)*255).astype(np.uint8))) 
            return '#%02x%02x%02x' % rgb_

        out_rgb = fq_c[1]
        out_hex = rgb_to_hex(colorsys.hsv_to_rgb(out_rgb[0], out_rgb[1], out_rgb[2]))

        return (out_rgb[0], out_rgb[1], out_rgb[2], out_hex)


iter_dir_4(tgtroot, cal_color_4)

import struct
rgbstr='605540'
struct.unpack('BBB',rgbstr.decode('hex'))

color_df = pd.read_csv(dataroot+'color4_boston.csv', index_col=0)

color_df.head(20)

ms_quantile = 0.12

w, h, d = original_shape = tuple(image.shape)
assert d == 3
color_array = np.array([np.array(struct.unpack('BBB',rgbstr[1:].decode('hex'))) for rgbstr in color_df['color'].dropna().values])

bandwidth = estimate_bandwidth(color_array, quantile=ms_quantile)
#ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(color_array)
ms = KMeans().fit(color_array)

labels = ms.predict(color_array)
labels_unique = np.unique(labels)
n_clusters = len(labels_unique)
centers = ms.cluster_centers_

def rgb_to_hex(rgb):
            rgb_ = tuple(((np.array(rgb)).astype(np.uint8))) 
            return '#%02x%02x%02x' % rgb_

centers_hex = [rgb_to_hex(i) for i in centers]
#centers_hex = [rgb_to_hex(colorsys.hsv_to_rgb(i[0],i[1],i[2])) for i in centers]
print centers_hex
 

rgb_to_hex([ 176.12280702,  171.80701754,  164.14619883])

color_array[labels==5]

labels[:20]

imarr = np.zeros([360,4*480,4]).astype(np.uint8) 
for i in range(4):
    indi_imarr = np.array(Image.open(tgtroot+'london/51.46136,-0.0926716179225_'+str(i)+'.png')) # 51.45386,-0.138461068985_
    imarr[:,i*480:(i+1)*480,:] = indi_imarr

alpha = imarr[:, :, 3]

mask = getmask(imarr[:,:,3], category=["Building", "SignSymbol", "Pole"])
base = (0*np.ones((360,480*4,3))).astype(np.uint8)
base[mask] = (imarr[:,:,:3])[mask]
color_array = (imarr[:,:,:3])[mask]
    
plt.figure(figsize=(18, 6))
plt.imshow(base) # plt.imshow(imarr)

# http://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html

def not_black(hsv):
    return hsv[2] > 0.001

majority_func = lambda s: s[2]*s[1][2]

ms_quantile = 0.12

image = Image.fromarray(base, 'RGB')
image = np.array(image, dtype=np.float64) / 255

# transform to a 2D numpy array.
w, h, d = original_shape = tuple(image.shape)
assert d == 3
image_array = np.reshape(image, (w * h, d))
image_array_sample = shuffle(image_array, random_state=0)[:1000]

bandwidth = estimate_bandwidth(image_array_sample, quantile=ms_quantile)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(image_array_sample)
labels = ms.predict(image_array)

#print "ms.cluster_centers_:", ms.cluster_centers_
labels_unique = np.unique(labels)
n_clusters = len(labels_unique)
#print 'n_clusters:', n_clusters
centers = ms.cluster_centers_
print centers

centers_hsv = [colorsys.rgb_to_hsv(i[0],i[1],i[2]) for i in centers]

# tuples of index, color, number of pixels with that color
sum_info = [(i, centers_hsv[i], np.sum(labels == i)) for i in range(len(centers_hsv))]
#print 'sum_info', sum_info
non_black = [i for i in sum_info if not_black(i[1])]
fq_c = sorted(sum_info , key=majority_func, reverse=True) # non_black
print "most frequent color:", fq_c

fq_id = fq_c[0][0]

X_s = [i[1] for i in fq_c]

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image
#print "labels:", np.unique(labels)

# Display all results, alongside original image
plt.figure(figsize=(20, 15))

plt.title('Quantized image (Mean-Shift)')

new_cookbook = centers
# new_cookbook = np.zeros([len(centers),3])
# new_cookbook[fq_id,:] = [ 0.45088474,  0.32654232,  0.26054519] #centers[fq_id]

print new_cookbook
print centers[fq_id], new_cookbook[fq_id,:]

plt.imshow(recreate_image(new_cookbook, labels, w, h))

lg_handles = []
lg_labels = []
def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
def rgb_to_hex(rgb):
    rgb_ = tuple(((np.array(rgb)*255).astype(np.uint8))) 
    # print '#%02x%02x%02x' % rgb_
    return '#%02x%02x%02x' % rgb_

for i in range(len(X_s)):
    hexi = rgb_to_hex(tuple(colorsys.hsv_to_rgb(X_s[i][0], X_s[i][1], X_s[i][2])))
    rect = mpatches.Patch(color=hexi)
    lg_handles.append(rect)
    lg_labels.append(str(i))
plt.legend(lg_handles,lg_labels,bbox_to_anchor=(0. ,1.09 ,1.,0.3),loc=8,
           ncol=5,mode='expand',borderaxespad=0,prop={'size':9},numpoints=1)





