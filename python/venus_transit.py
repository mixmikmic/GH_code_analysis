get_ipython().magic('matplotlib inline')
from IPython.display import IFrame
IFrame(src='https://stanford.edu/~mbobra/venus/venus_transit.mp4',width=400,height=600)

import json, urllib, numpy as np, matplotlib.pylab as plt, matplotlib.ticker as mtick
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from datetime import datetime as dt_obj
from astropy.io import fits
from matplotlib.dates import *
import matplotlib.gridspec as gridspec
from sunpy.cm import color_tables as ct
from scipy import signal
get_ipython().magic("config InlineBackend.figure_format='retina'")

url = "http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info?ds=hmi.Ic_45s[2012.06.03_22:00:59-2012.06.07_22:00:59]&op=rs_list&key=T_REC,DATAMEA2"
response = urllib.urlopen(url)
data = json.loads(response.read())

datamea2 = [] # this holds the keyword DATAMEA2
t_rec  = [] # this holds the keyword T_REC

n_elements = len(data['keywords'][0]['values'])

count = 0.0
for i in range(n_elements):
    t_rec.append(data['keywords'][0]['values'][i])
    if 'MISSING' in str(data['keywords'][1]['values'][i]):
        print 'missing datamea2 value at time ',data['keywords'][0]['values'][i]
        datamea2_pre = data['keywords'][1]['values'][i-1]
        if (datamea2_pre != 'MISSING'):
            datamea2.append(data['keywords'][1]['values'][i-1])
            print 'substituting',data['keywords'][1]['values'][i-1]
        else:
            datamea2.append(data['keywords'][1]['values'][i-3])
            print 'substituting',data['keywords'][1]['values'][i-3]
        count = count + 1.0
    else:
        datamea2.append(data['keywords'][1]['values'][i])
datamea2 = np.array(datamea2).astype(float)
t_rec = np.array(t_rec)
print 'there are ',count,'missing datamea2 values'
print 'the length of datamea2 and t_rec are',datamea2.shape,t_rec.shape

chunk = []
for i in range(1920, n_elements-1921, 1920):
    before_chunk = datamea2[i-1920:i]
    after_chunk = datamea2[i+1920:i+3840]
    avg_chunk = (before_chunk + after_chunk) / 2.0
    chunk.append(datamea2[i:i+1920] - avg_chunk)
    print 'chunk',i

datamea2 = np.array(chunk).flatten()
t_rec = t_rec[1920:1920+1920+1920]
print datamea2.shape, t_rec.shape

def parse_tai_string(tstr,datetime=True):
    year   = int(tstr[:4])
    month  = int(tstr[5:7])
    day    = int(tstr[8:10])
    hour   = int(tstr[11:13])
    minute = int(tstr[14:16])
    second = int(tstr[17:19])
    if datetime: return dt_obj(year,month,day,hour,minute,second)
    else: return year,month,day,hour,minute,second

x = np.array([parse_tai_string(t_rec[i],datetime=True) for i in range(t_rec.size)])

times_transit = x[1900:2456]
datamea2_transit = datamea2[1900:2456]

url = 'http://jsoc.stanford.edu/data/events/Venus_HMI_Ic/WholeSun/list.body'
response = urllib.urlopen(url)
times = response.read()
times = times.split('},') # then split it into lines

image_times = []
for i in range(0):
    image_times.append(times[i][17:36])
for i in range(1,11):
    image_times.append(times[i][18:37])
for i in range(12,101):
    image_times.append(times[i][19:38])
for i in range(102,183):
    image_times.append(times[i][20:39])
image_times = np.array(image_times)

image_times[0] = '2012-06-05_21:46:00'
image_times[1] = '2012-06-05_21:53:00'

x_image_times = np.array([parse_tai_string(image_times[i],datetime=True) for i in range(image_times.size)])

imagetimedelta = []
for i in range(x_image_times.size-1):
    imagetimedelta.append((x_image_times[i+1] - x_image_times[i]).total_seconds())
imagetimedelta = np.array(imagetimedelta)

imagetimedelta = (np.round((imagetimedelta/45.+0.1))).astype(int)

get_ipython().run_cell_magic('capture', '', 'count = 0\nfor i in range(imagetimedelta.shape[0]): \n    count = imagetimedelta[i] + count \n    j = i + 1\n    counter = "%04d"%j\n    if (counter == \'0024\'):   # this image is messed up\n        continue\n    url = "http://jsoc.stanford.edu/data/events/Venus_HMI_Ic/WholeSun/images/"+counter+".jpg"\n    data = urllib.urlretrieve(url)\n    image = mpimg.imread(data[0])\n\n    fig = plt.figure(figsize=(10,10),facecolor=\'black\')\n    ax_image = plt.subplot2grid((7, 5), (0, 0), colspan=5, rowspan=5)\n    ax_image.get_xaxis().set_ticks([])\n    ax_image.get_yaxis().set_ticks([])\n    ax_image.set_axis_bgcolor(\'black\')\n    plt.imshow(image[:-30,30:])\n\n    #ax = fig.add_subplot(2,1,1)\n    ax = plt.subplot2grid((7, 5), (5, 1), colspan=3, rowspan=2)\n    #fig, ax = plt.subplots(figsize=(5,5))           # define the size of the figure\n    cornblue = (100/255., 149/255., 147/255., 1.0)   # create a cornflower blue color\n    grey     = (211/255., 211/255., 211/255., 1.0)   # create a light grey color\n\n    # define some style elements\n    marker_style = dict(linestyle=\'\', markersize=8, fillstyle=\'full\',color=cornblue,markeredgecolor=cornblue)\n    background_marker_style = dict(linestyle=\'\', markersize=8, fillstyle=\'full\',color=grey,markeredgecolor=grey)\n    text_style = dict(fontsize=16, fontdict={\'family\': \'helvetica\'}, color=grey)\n    ax.tick_params(labelsize=14)\n    ax.spines[\'bottom\'].set_color(\'grey\')\n    ax.spines[\'left\'].set_color(\'grey\')\n    ax.spines[\'bottom\'].set_linewidth(3)\n    ax.spines[\'left\'].set_linewidth(3)\n    \n    # ascribe the data to the axes\n    ax.plot(times_transit[:-1], datamea2_transit[:-1],\'o\',**background_marker_style)\n    ax.plot(times_transit[0:count], datamea2_transit[0:count],\'o\',**marker_style)\n\n    # format the x-axis with universal time\n    locator = AutoDateLocator()\n    locator.intervald[HOURLY] = [2] # only show every 6 hours\n    formatter = DateFormatter(\'%H\')\n    ax.xaxis.set_major_locator(locator)\n    ax.xaxis.set_major_formatter(formatter)\n    ax.set_xlabel(\'time\',**text_style)\n\n    # set the x and y axis ranges\n    ax.set_xlim([times_transit[0],x[2465]])\n    ax.set_ylim([-80,20])\n\n    # label the axes and the plot\n    ax.get_yaxis().set_ticks([])\n    ax.get_xaxis().set_ticks([])\n    ax.set_ylabel(\'brightness\',**text_style)\n    ax.xaxis.labelpad=5\n    ax.yaxis.labelpad=5\n    ax.set_axis_bgcolor(\'black\')\n    fig.subplots_adjust()\n    fig.savefig(counter+\'.jpg\',bbox_inches=\'tight\',facecolor=fig.get_facecolor(), transparent=True, dpi=192)')

