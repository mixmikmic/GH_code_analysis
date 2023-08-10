import sys
sys.path.insert(0, '/Users/rb/Google_Drive/Waterloo/projects/cancer_hist/src')
from functions import*
from heat_models import*

import pickle
from keras.models import load_model
import os
import numpy as np
import pandas as pd
import glob
import random
from PIL import Image
import matplotlib.pyplot as plt

test_folder = '/Users/rb/Documents/waterloo/projects/cancer_hist/final/full_slides/valid'
heat_folder = '/Users/rb/Documents/waterloo/projects/cancer_hist/final/heat_conv_incp3_192/valid'
cutoff=.95
radius = 5

acc=test_heat_preds(test_folder=test_folder, heat_folder=heat_folder, radius=radius, cutoff=cutoff, output_class=True, stride=2)

TP = len(acc["all_matched_pts"])/float(acc["total_nuclei"])
FPT = (float(acc["num_predicted"])-len(acc["all_matched_pts"]))/float(acc["total_nuclei"])
print TP
print FPT

import sys
sys.path.insert(0, '/Users/rb/Google_Drive/Waterloo/projects/cancer_hist/src')
from functions import*
import numpy as np
import glob
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
import pprint
get_ipython().run_line_magic('matplotlib', 'inline')

weird_list=[]

train_dir = '/Users/rb/Documents/waterloo/projects/cancer_hist/final/full_slides/train'
all_files=glob.glob(os.path.join(train_dir, '*'))
print 'len(all_files)', len(all_files)
all_xml = [loc for loc in all_files if 'key' in loc]
print 'len(all_xml)', len(all_xml)
all_min_dists = []
for xml_loc in all_xml:
    all_points = get_points_xml(xml_loc)
    point_list = all_points[:, 0:2]
    for index, point in enumerate(point_list):
        temp_point_list = np.delete(point_list, (index), axis=0) # don't match the point with itself
        dists = np.sqrt(np.sum((temp_point_list - point) ** 2, axis=1))
        min_ind = np.argmin(dists) 
        all_min_dists.append(dists[min_ind])
        if dists[min_ind]<4:
            weird_list.append(xml_loc) #.rsplit('/', 1)[-1])

print Counter(weird_list)
weird_list = list(set(weird_list))
print len(weird_list)

plt.hist(all_min_dists, bins=30, range=[0, 30], normed=False)
plt.title('Distance between cell and nearest neighbour')
plt.ylabel('Probability');

test_folder = '/Users/rb/Documents/waterloo/projects/cancer_hist/final/full_slides/valid'
heat_folder = '/Users/rb/Documents/waterloo/projects/cancer_hist/final/heat_conv_incp3_192/valid'
abs_error_means=[]
settings=[]
results = np.zeros((11, 4, 2))
for col, radius in enumerate(range(4, 8)):
    print radius
    for row, cutoff in enumerate([.3, .5, .7, .8, .9, .95, .97, .98, .99, .999, .9999]):
        acc = test_heat_preds(test_folder=test_folder, heat_folder=heat_folder, radius=radius, cutoff=cutoff, output_class=True, stride = 2)
        TP = len(acc["all_matched_pts"])/float(acc["total_nuclei"])
        FPT = (float(acc["num_predicted"])-len(acc["all_matched_pts"]))/float(acc["total_nuclei"])
        results[row, col, 0] = TP
        results[row, col, 1] = FPT
        

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from matplotlib import rc


plt.plot(results[:,0,1], results[:,0,0], linewidth=2.0, label=r'$rad = 4 \mu m$')
plt.plot(results[:,1,1], results[:,1,0], linewidth=2.0, label=r'$rad = 5 \mu m$')
plt.plot(results[:,2,1], results[:,2,0], linewidth=2.0, label=r'$rad = 6 \mu m$')
plt.plot(results[:,3,1], results[:,3,0], linewidth=2.0, label=r'$rad = 7 \mu m$')
plt.legend()
axes = plt.gca()
axes.set_xlim([.35, .8])
# axes.set_ylim([.7, 1])
axes.set_xlabel("False Positives / Total Nuclei")
axes.set_ylabel("True Positive Rate")

plt.title('Radius vs. Performance') 

plt.show()

train_dir = '/Users/rb/Documents/waterloo/projects/cancer_hist/final/full_slides/train'

all_files=glob.glob(os.path.join(train_dir, '*'))
print 'len(all_files)', len(all_files)
all_xml = [loc for loc in all_files if 'key' in loc]
print 'len(all_xml)', len(all_xml)
lymphocyte_dists = []
normal_dists = []
malignant_dists = []

for xml_loc in all_xml:
    all_points = get_points_xml(xml_loc)
    point_list = all_points[:, 0:2]
    for index, point in enumerate(point_list):
        temp_point_list = np.delete(point_list, (index), axis=0) # don't match the point with itself
        dists = np.sqrt(np.sum((temp_point_list - point) ** 2, axis=1))
        min_ind = np.argmin(dists) 
        if (all_points[index, 2] == 1) :
            lymphocyte_dists.append(dists[min_ind])
        elif (all_points[index, 2] == 2):
            normal_dists.append(dists[min_ind])
        elif (all_points[index, 2] == 3):
            malignant_dists.append(dists[min_ind])
        else:
            print 'error'

print 'len(lymphocyte_dists)', len(lymphocyte_dists)
print 'len(normal_dists)', len(normal_dists)
print 'len(malignant_dists)', len(malignant_dists)

plt.hist(lymphocyte_dists, bins=30, range=[0, 30], normed=False)
plt.title('Distance between lymphocyte cell and nearest neighbour')
plt.ylabel('Probability');
plt.show()

plt.hist(normal_dists, bins=30, range=[0, 30], normed=False)
plt.title('Distance between normal cell and nearest neighbour')
plt.ylabel('Probability');
plt.show()

plt.hist(malignant_dists, bins=30, range=[0, 30], normed=False)
plt.title('Distance between malignant cell and nearest neighbour')
plt.ylabel('Probability');
plt.show()

import sys
sys.path.insert(0, '/Users/rb/Google_Drive/Waterloo/projects/cancer_hist/src')
from functions import*
from heat_models import*

test_folder = '/Users/rb/Documents/waterloo/projects/cancer_hist/final/full_slides/valid'
heat_folder = '/Users/rb/Documents/waterloo/projects/cancer_hist/final/heat_conv_incp3_192/valid'
abs_error_means=[]
settings=[]
results = np.zeros((12, 5, 2)) 
radius_list = [(5,5,5), (5, 5, 6), (4, 5, 6), (5, 6, 7), (6, 6, 6)]
for col, radius in enumerate(radius_list):
    print radius
    for row, cutoff in enumerate([.2, .5, .7, .8, .9, .95, .97, .98, .99, .999, .9999, .99999]):
        acc = test_heat_preds(test_folder=test_folder, heat_folder=heat_folder, radius=radius, cutoff=cutoff, output_class=True, stride=2)
        TP = len(acc["all_matched_pts"])/float(acc["total_nuclei"])
        FPT = (float(acc["num_predicted"])-len(acc["all_matched_pts"]))/float(acc["total_nuclei"])
        results[row, col, 0] = TP
        results[row, col, 1] = FPT

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.plot(results[:,0,1], results[:,0,0], linewidth=2.0, label=r'$rad = 5, 5, 5 \mu m$')
plt.plot(results[:,1,1], results[:,1,0], linewidth=2.0, label=r'$rad = 5, 5, 6 \mu m$')
plt.plot(results[:,2,1], results[:,2,0], linewidth=2.0, label=r'$rad = 4, 5, 6 \mu m$')
plt.plot(results[:,3,1], results[:,3,0], linewidth=2.0, label=r'$rad = 5, 6, 7 \mu m$')
plt.plot(results[:,4,1], results[:,4,0], linewidth=2.0, label=r'$rad = 6, 6, 6 \mu m$')

plt.legend()
plt.legend()
axes = plt.gca()
axes.set_xlabel("False Positives / Total Nuclei")
axes.set_ylabel("True Positive Rate")
axes.set_xlim([.3, .6])
# axes.set_ylim([.7, 1])

plt.title('Radius vs. Performance')

plt.show()





