# Download images
# Mostly code from https://github.com/rezoo/imagenet-python

import sys
import os
from xml.etree import ElementTree
import requests
import json

BASE_URL = "http://www.image-net.org/download/synset"
USERNAME = "lqkhoo"
ACCESS_KEY = "2f9264606df44886088c60983ca4ec45cb8e62c9"

BASE_PATH = "data/images/tar/"

synsets = None
NAMES_FILE_PATH = 'data/named_populated_synsets.json'
with open(NAMES_FILE_PATH) as f:
    synsets = json.load(f)

print(len(synsets.keys()))
    
for synset in synsets:
    params = {
        "wnid": synset,
        "username": USERNAME,
        "accesskey": ACCESS_KEY,
        "release": "latest",
        "src": "stanford"
    }
    
    print("Downloading images for " + synset + " (" + synsets[synset] + ")")
    write_path = BASE_PATH + synset + ".tar"
    if not os.path.exists(write_path) or os.path.getsize(write_path) == 0:
        response = requests.get(BASE_URL, params=params)
        content_type = response.headers["content-type"]
        if content_type.startswith("text"):
            print("  WARNING: 404 error downloading synset" + synset)
        else:
            with open(write_path, "wb") as f:
                f.write(response.content)
    
    else:
        print("  Images already downloaded. Moving on...")
            
print("All done.")

# Extract images to /raw

import os
import tarfile
from os import listdir
from os.path import isfile, join
import json
import numpy as np

INPUT_DIR = "data/images/tar/"
OUTPUT_DIR = "data/images/raw/"

files = [f for f in listdir(INPUT_DIR) if isfile(join(INPUT_DIR, f))]

# load synsets
FILE_PATH_1 = 'data/final_synsets_counts.json'
with open('data/final_synsets_counts.json') as f:
    synsets = json.load(f)
    f.close()
    
for synset in synsets:   
    input_dir = join(INPUT_DIR, synset + ".tar")
    output_dir = join(OUTPUT_DIR, synset)
    tar = tarfile.open(input_dir)
    
    # Extract to /raw
    print("Extracting " + synset)
    tar.extractall(path=output_dir)

print("All done.")

# Filter. We take images which meet the following criteria:
#   at least 256x256 pixels
#   3-channel RGB. We discard black & white, 4-channel CMYK jpegs, or other formats 
#      that do not have the shape (>=224, >=224, 3)
# Dump these images to /filtered

import os
from os import listdir
from os.path import isfile, join
from shutil import copyfile
from PIL import Image

INPUT_DIR = "data/images/raw/"
OUTPUT_DIR = "data/images/filtered/"

dirs = [d for d in listdir(INPUT_DIR) if not isfile(join(INPUT_DIR, d))]
for i in range(len(dirs)):
    d = dirs[i]
    print("Processing: " + d + " (" + str(i+1) + "/" + str(len(dirs)) + ")")
    
    dirpath = join(INPUT_DIR, d)
    files = [f for f in listdir(dirpath) if isfile(join(dirpath, f))]
    num_files = len(files)
    num_ignored_files = 0
    for f in files:
        file_path = join(dirpath, f)
        img = Image.open(file_path)
        img.load()
        data = np.asarray(img)
        if len(data.shape) != 3:
            # print("    Ignored file: Incorrect no. of dimensions." + str(data.shape))
            num_ignored_files += 1
            continue
        elif data.shape[2] != 3:
            # print("    Ignored file: Incorrect no. of channels." + str(data.shape))
            num_ignored_files += 1
            continue
        elif data.shape[0] < 224 or data.shape[1] < 224:
            # print("    Ignored file: Image smaller than 100px." + str(data.shape))
            num_ignored_files += 1
            continue
            
        output_dirpath = join(OUTPUT_DIR, d)
        if not os.path.exists(output_dirpath):
            os.makedirs(output_dirpath)
            
        copyfile(join(dirpath, f), join(output_dirpath, f))
    print("  " + str(num_files - num_ignored_files) + " / " + str(num_files) + " passed filter.")
            
print("All done")

# Count and sort synsets

import os
from os import listdir
from os.path import isfile, join
import json
from pprint import pprint

INPUT_DIR = "data/images/filtered/"
counts = {}

dirs = [d for d in listdir(INPUT_DIR) if not isfile(join(INPUT_DIR, d))]
for i in range(len(dirs)):
    d = dirs[i]
    print("Processing: " + d + " (" + str(i+1) + "/" + str(len(dirs)) + ")")
    
    dirpath = join(INPUT_DIR, d)
    files = [f for f in listdir(dirpath) if isfile(join(dirpath, f))]
    num_files = len(files)
    counts[d] = num_files
    
sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
sorted_counts = [t for t in sorted_counts if t[1] >= 600]

synsets = {}
synsets.update(sorted_counts)
pprint(len(synsets.items()))

with open("data/filtered_synsets.json", "w") as f:
    json.dump(synsets, f)

# Now we need to ensure each image has a bounding box. We filter again.
import json
from shutil import copyfile

INPUT_DIR = "data/images/filtered/"
OUTPUT_DIR = "data/images/bboxfiltered/"
JSON_BBOX_DIR = "data/bbox/json/"

dirs = [d.split('.')[0] for d in listdir(JSON_BBOX_DIR) if isfile(join(JSON_BBOX_DIR, d))]
for i in range(len(dirs)):
    d = dirs[i]
    with open(join(JSON_BBOX_DIR, d + ".json")) as f:
        bbox_dict = json.load(f)
    
    print("Processing: " + d + " (" + str(i+1) + "/" + str(len(dirs)) + ")")
    
    dirpath = join(INPUT_DIR, d)
    files = [f for f in listdir(dirpath) if isfile(join(dirpath, f))]
    
    for f in files:
        filepath = join(dirpath, f)
        output_dirpath = join(OUTPUT_DIR, d)
        if not os.path.exists(output_dirpath):
            os.makedirs(output_dirpath)
        if f in bbox_dict:
            copyfile(filepath, join(output_dirpath, f))
            
print("All done.")

# Count images again
import numpy as np
from os import listdir
from os.path import isfile, join
from pprint import pprint

INPUT_DIR = "data/images/bboxfiltered"

dirs = [d for d in listdir(INPUT_DIR) if not isfile(join(INPUT_DIR, d))]

counts = {}
for i in range(len(dirs)):
    d = dirs[i]
    dirpath = join(INPUT_DIR, d)
    files = [f for f in listdir(dirpath) if isfile(join(dirpath, f))]
    count = len(files)
    counts[d] = count
            
pprint(counts)
n = len([c for c in counts.values() if c >= 400])
total = np.sum(np.array([c for c in counts.values() if c >= 400]))
print(n)
print(total)

# Create training, test, and validation splits. 
# We pick out 160 synsets that contain the most images and make the splits on that set
# We hold out ALL instances in the remaining 38 synsets to test for generalization

import json
import os
from os import listdir
from os.path import isfile, join
from shutil import copyfile
import numpy as np

SYNSETS_FILE = "data/filtered_synsets.json"
INPUT_DIR = "data/images/bboxfiltered/"
OUTPUT_DIR = "data/images/"
counts = {}

with open(SYNSETS_FILE) as f:
    synsets = json.load(f)
    print(len(synsets.items()))

counts = {}
dirs = [d for d in listdir(INPUT_DIR) if not isfile(join(INPUT_DIR, d))]
# Sort by no. of files in each dir
for d in dirs:
    dirpath = join(INPUT_DIR, d)
    files = [f for f in listdir(dirpath) if isfile(join(dirpath, f))]
    num_files = len(files)
    if num_files >= 400:
        counts[d] = num_files
    
sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
print(len(sorted_counts))

synsets_to_be_split = list(map(lambda x: x[0], sorted_counts[:160]))
synsets_to_be_held_out = list(map(lambda x: x[0], sorted_counts[160:]))
print(len(synsets_to_be_split))
print(len(synsets_to_be_held_out))


for d in synsets_to_be_held_out:
    print("Processing holdout: " + d)
    dirpath = join(INPUT_DIR, d)
    holdout_output_dir = join(OUTPUT_DIR, "holdout")
    
    holdout_train_output_dir = join(holdout_output_dir, "train", d)
    holdout_val_output_dir = join(holdout_output_dir, "val", d)
    holdout_test_output_dir = join(holdout_output_dir, "test", d)
    
    if not os.path.exists(holdout_train_output_dir): os.makedirs(holdout_train_output_dir)
    if not os.path.exists(holdout_val_output_dir): os.makedirs(holdout_val_output_dir)
    if not os.path.exists(holdout_test_output_dir): os.makedirs(holdout_test_output_dir)
    
    files = [f for f in listdir(dirpath) if isfile(join(dirpath, f))]
    num_files = len(files)
    val = files[0:50]
    test = files[51:100]
    train = files[101:]
    # train, val, test = np.split(files, [int((10.0/12.0)*num_files), int((11.0/12.0)*num_files)])
    
    for f in train:
        copyfile(join(dirpath, f), join(holdout_train_output_dir, f))
    for f in val:
        copyfile(join(dirpath, f), join(holdout_val_output_dir, f))
    for f in test:
        copyfile(join(dirpath, f), join(holdout_test_output_dir, f))


for i in range(len(synsets_to_be_split)):
    
    d = synsets_to_be_split[i]
    print("Processing: " + d + " (" + str(i+1) + "/" + str(len(synsets_to_be_split)) + ")")
    if d not in synsets:
        print("  Insufficient no. of images. Ignoring synset.")
        continue
    
    dirpath = join(INPUT_DIR, d)
    
    train_output_dir = join(OUTPUT_DIR, "train", d)
    val_output_dir = join(OUTPUT_DIR, "val", d)
    test_output_dir = join(OUTPUT_DIR, "test", d)
    if not os.path.exists(train_output_dir): os.makedirs(train_output_dir)
    if not os.path.exists(val_output_dir): os.makedirs(val_output_dir)
    if not os.path.exists(test_output_dir): os.makedirs(test_output_dir)
    
    files = [f for f in listdir(dirpath) if isfile(join(dirpath, f))]
    num_files = len(files)
    val = files[0:50]
    test = files[51:100]
    train = files[101:]
    # train, val, test = np.split(files, [int((10.0/12.0)*num_files), int((11.0/12.0)*num_files)])
    
    for f in train:
        copyfile(join(dirpath, f), join(train_output_dir, f))
    for f in val:
        copyfile(join(dirpath, f), join(val_output_dir, f))
    for f in test:
        copyfile(join(dirpath, f), join(test_output_dir, f))


print("All done.")



