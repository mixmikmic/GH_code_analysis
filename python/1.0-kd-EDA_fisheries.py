import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.misc import imread
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

from subprocess import check_output
print(check_output(["ls", "/home/ubuntu/data/fishing/train"]).decode("utf8"))

sub_folders = check_output(["ls", "/home/ubuntu/data/fishing/train/"]).decode("utf8").strip().split('\n')
count_dict = {}
for sub_folder in sub_folders:
    num_of_files = len(check_output(["ls", "/home/ubuntu/data/fishing/train/"+sub_folder]).decode("utf8").strip().split('\n'))
    print("Number of files for the species",sub_folder,":",num_of_files)
    count_dict[sub_folder] = num_of_files
    
plt.figure(figsize=(12,4))
sns.barplot(list(count_dict.keys()), list(count_dict.values()), alpha=0.8)
plt.xlabel('Fish Species', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.show()

import glob
import PIL

images = [ PIL.Image.open(f) for f in glob.glob('/home/ubuntu/data/fishing/train/LAG/*') ]
filenames = [ f for f in glob.glob('/home/ubuntu/data/fishing/train/LAG/*') ]

def img2array(im):
    if im.mode != 'RGB':
        im = im.convert(mode='RGB')
    return np.fromstring(im.tobytes(), dtype='uint8').reshape((im.size[1], im.size[0], 3))

np_images = [ img2array(im) for im in images ]

for (img, filen) in zip(np_images, filenames):
    plt.figure()
    plt.imshow(img)
    plt.title("filename: "+filen, fontsize=15)



