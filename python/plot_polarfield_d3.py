import json, urllib, numpy as np, matplotlib.pylab as plt, matplotlib.ticker as mtick, mpld3
from mpld3 import plugins
from datetime import datetime as dt_obj
from matplotlib.dates import *
get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format='retina'")
mpld3.enable_notebook()

# first get the most recent timestamp, or t_rec
url = "http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info?ds=hmi.meanpf_720s[$]&op=rs_list&key=T_REC"
response = urllib.urlopen(url)
data = json.loads(response.read())
t_last = data['keywords'][0]['values'][0]

# now gather all the data between SDO launch and the most recent t_rec
url = "http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info?ds=hmi.meanpf_720s[2010.05.01_00_TAI-"+t_last+"@12h]&op=rs_list&key=T_REC,CAPN2,CAPS2"
response = urllib.urlopen(url)
data = json.loads(response.read())

capn2 = [] # this holds the keyword CAPN2 (Mean radial field in N60-N90) 
caps2 = [] # this holds the keyword CAPS2 (Mean radial field in S60-S90)
t_rec  = [] # this holds the keyword T_REC

n_elements = len(data['keywords'][1]['values'])

for i in range(n_elements):
    capn2.append(float(data['keywords'][1]['values'][i]))

for i in range(n_elements):
    caps2.append(float(data['keywords'][2]['values'][i]))

for i in range(n_elements):
    t_rec.append(data['keywords'][0]['values'][i])

capn2 = np.array(capn2)
caps2 = np.array(caps2)
t_rec  = np.array(t_rec,dtype='S16') # this dtype is ok to set as the format of T_REC will never change

err_capn2 = np.zeros(n_elements)
err_caps2 = np.zeros(n_elements)

for i in range(0,n_elements):
    err_capn2[i] = np.std(capn2[i-30:i+30])
    
for i in range(0,n_elements):
    err_caps2[i] = np.std(caps2[i-30:i+30])

def parse_tai_string(tstr,datetime=True):
    year   = int(tstr[:4])
    month  = int(tstr[5:7])
    day    = int(tstr[8:10])
    hour   = int(tstr[11:13])
    minute = int(tstr[14:16])
    if datetime: return dt_obj(year,month,day,hour,minute)
    else: return year,month,day,hour,minute

x = np.array([parse_tai_string(t_rec[i],datetime=True) for i in range(t_rec.size)])

fig, ax = plt.subplots(figsize=(10,8))      # define the size of the figure
orangered = (1.0,0.27,0,1.0)                # create an orange-red color
cornblue  = (0.39,0.58,0.93,1.0)            # create an cornflower blue color

# define some style elements
marker_style = dict(linestyle='-', linewidth=4, fillstyle='full',color=orangered,markeredgecolor=orangered)
marker_style2 = dict(linestyle='-', linewidth=4, fillstyle='full',color=cornblue,markeredgecolor=cornblue)
text_style = {'family' : 'cursive','color'  : 'black','weight' : 'normal','size'   : 16}
#text_style = dict(fontsize=16, fontdict={'family': 'georgia'})

ax.tick_params(labelsize=14)
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%1f'))

# ascribe the data to the axes
ax.plot(x, capn2, label='north pole', **marker_style)
ax.plot(x, caps2, label='south pole', **marker_style2)
ax.set_ylim([-4,4])

# add the error bars
ax.fill_between(x, capn2-err_capn2, capn2+err_capn2, facecolor=orangered, edgecolor='None', alpha=0.4, interpolate=True)
ax.fill_between(x, caps2-err_caps2, caps2+err_caps2, facecolor=cornblue, edgecolor='None', alpha=0.4, interpolate=True)

# format the x-axis with universal time
locator = AutoDateLocator()
locator.intervald[DAILY] = [1] # only show every year
formatter = DateFormatter('%Y')
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

# label the axes and the plot
ax.set_xlabel('time', fontdict = text_style)
ax.set_ylabel('mean radial field strength in gauss', fontdict=text_style, labelpad = -40)
ax.set_title('mean polar field', fontdict = text_style) # annotate the plot with a start time
fig.set_size_inches(10,4)
legend = plt.legend(loc='upper right', fontsize=12, framealpha=0.0,title='')
legend.get_frame().set_linewidth(0.0)

mpld3.save_html(fig,'12hcadence_nointerp.html')

avg_capn2 = np.zeros(n_elements)
avg_caps2 = np.zeros(n_elements)

for i in range(0,n_elements):
    avg_capn2[i] = np.mean(capn2[i-30:i+30])
    
for i in range(0,n_elements):
    avg_caps2[i] = np.mean(caps2[i-30:i+30])

fig, ax = plt.subplots(figsize=(10,8))      # define the size of the figure
orangered = (1.0,0.27,0,1.0)                # create an orange-red color
cornblue  = (0.39,0.58,0.93,1.0)            # create an cornflower blue color

# define some style elements
marker_style = dict(linestyle='-', linewidth=4, fillstyle='full',color=orangered,markeredgecolor=orangered)
marker_style2 = dict(linestyle='-', linewidth=4, fillstyle='full',color=cornblue,markeredgecolor=cornblue)
text_style = {'family' : 'cursive','color'  : 'black','weight' : 'normal','size'   : 16}
#text_style = dict(fontsize=16, fontdict={'family': 'georgia'})

ax.tick_params(labelsize=14)
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%1f'))

# ascribe the data to the axes
ax.plot(x, avg_capn2, label='north pole', **marker_style)
ax.plot(x, avg_caps2, label='south pole', **marker_style2)
ax.set_ylim([-4,4])

# add the error bars
ax.fill_between(x, avg_capn2-err_capn2, avg_capn2+err_capn2, facecolor=orangered, edgecolor='None', alpha=0.4, interpolate=True)
ax.fill_between(x, avg_caps2-err_caps2, avg_caps2+err_caps2, facecolor=cornblue, edgecolor='None', alpha=0.4, interpolate=True)

# format the x-axis with universal time
locator = AutoDateLocator()
locator.intervald[DAILY] = [1] # only show every year
formatter = DateFormatter('%Y')
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

# label the axes and the plot
ax.set_xlabel('time', fontdict = text_style)
ax.set_ylabel('mean radial field strength in gauss', fontdict=text_style, labelpad = -40)
ax.set_title('mean polar field', fontdict = text_style) # annotate the plot with a start time
fig.set_size_inches(10,4)
legend = plt.legend(loc='upper right', fontsize=12, framealpha=0.0,title='')
legend.get_frame().set_linewidth(0.0)

mpld3.save_html(fig,'12hcadence_interp.html')

