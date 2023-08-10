#Rename the files downloaded from TCGA

import os
import shutil

list_not_dir = ['gdc_manifest_20170512_072412.txt', 'test.svs', 'y_train.txt', 'gdc-client']
for filename in os.listdir("/home/cedoz/data/"):
    if filename not in list_not_dir:
        if filename[-3:] != "svs":
            my_path = "/home/cedoz/data/%s"%filename
            svs_image = [f for f in os.listdir(my_path) if os.path.isfile(os.path.join(my_path, f)) if f != "annotations.txt"]
            os.chdir(my_path)
            if svs_image != []:
                if len(str(svs_image[0]>12)):
                    os.rename(svs_image[0], svs_image[0][:12] + ".svs")
                    shutil.move(svs_image[0][:12] + ".svs", os.path.join("/home/cedoz/data/", svs_image[0][:12] + ".svs"))
            shutil.rmtree(my_path)
os.chdir("/home/cedoz/")

import numpy as np

y_train = {}
with open("data/y_train.txt") as f:
    for line in f:
        (key, val) = line.split()
        key = str(key).replace('.','-')
        y_train[key] = int(val)

svs_images = []
for filename in os.listdir("/home/cedoz/data/"):
    if filename not in list_not_dir:
        svs_images.append(filename[:-4])

svs_images
y_train.keys()
intersect_samples = [val for val in svs_images if val in y_train.keys()]

print len(intersect_samples)
print len(svs_images)
print len(y_train.keys())

svs_images



