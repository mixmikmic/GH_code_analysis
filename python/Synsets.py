## Get number of synsets which have bounding box annotations, which also contain >= 600 images

import os, os.path
from pprint import pprint
import json
import numpy as np

bbox_annotations_path = 'data/bbox/Annotation'
MIN_NO_OF_IMAGES = 600
counts = {}

synsets = [f for f in os.listdir(bbox_annotations_path)
                if not os.path.isfile(os.path.join(bbox_annotations_path, f))]

for synset in synsets:
    path = bbox_annotations_path + '/' + synset
    count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    counts[synset] = count

populated_synsets = {k: v for k, v in counts.items() if v >= MIN_NO_OF_IMAGES}

print(len(populated_synsets.items()))
print(populated_synsets)

## Now make sure those synsets have names so they are interpretable -- because some do not
## Discard the unnamed synsets

synset_names = None
NAMES_FILE_PATH = 'data/synset_names.json'
with open(NAMES_FILE_PATH) as f:
    synset_names = json.load(f)
    
named_populated_synsets = {}
discarded_count = 0

for synset in populated_synsets:
    if synset in synset_names.keys():
        synset_name = synset_names[synset].split(',')[0] # Only take the first of comma-separated alternative names
        named_populated_synsets[synset] = synset_name
    else:
        discarded_count += 1
        
print(named_populated_synsets)

print("Remaining: " + str(len(named_populated_synsets.items())))
print("Discarded: " + str(discarded_count))


# Write to file
FILE_PATH = 'data/named_populated_synsets.json'
data = json.dumps(named_populated_synsets)
with open(FILE_PATH, 'w') as f:
    f.write(data)


final_synsets_counts = {}
final_synsets_names = {}
synsets_sorted = sorted(named_populated_synsets.items(), key=lambda x: x[1], reverse=True)
print(len(synsets_sorted))
# for i in range(200):
for i in range(len(synsets_sorted)):
    synset = synsets_sorted[i][0]
    if synset in named_populated_synsets:
        final_synsets_names[synset] = named_populated_synsets[synset]
        final_synsets_counts[synset] = synsets_sorted[i][1]

        
FILE_PATH_1 = 'data/final_synsets_counts.json'
FILE_PATH_2 = 'data/final_synsets_names.json'
data1 = json.dumps(final_synsets_counts)
data2 = json.dumps(final_synsets_names)
with open(FILE_PATH_1, 'w') as f:
    f.write(data1)
with open(FILE_PATH_2, 'w') as f:
    f.write(data2)

print("All done.")



