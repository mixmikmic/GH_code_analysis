import json
import pickle
import os
import numpy as np

DATASET_PATH = '/home/luoyy/datasets_large/mscoco/coco/'
ANNOTATIONS = os.path.join(DATASET_PATH, 'annotations/')
VAL_ANN = os.path.join(ANNOTATIONS, 'instances_val2014.json')
TRAIN_VAL = os.path.join(ANNOTATIONS, 'instances_train2014.json')

# image_id: file_name mapping
def imid_fn(image_dict):
    """
        image_dict: json['images'], from annotations json
    """
    imid_fn = {}
    for img  in image_dict:
        imid = img['id']
        ifn = img['file_name']
        imid_fn[imid] = ifn
    return imid_fn

from collections import defaultdict
def cat_vector(ann_dict, imid_fn):
    """
    Args:
        ann_dict: j['annotations']
        imid_fn : map from imid to filename
    Returns vector, consisting of objects, represented on image
    """
    cv_dict = defaultdict(list)
    for ann in ann_dict:
        ann_imid = ann['image_id']
        f_name = imid_fn[ann_imid]
        cat_id = ann['category_id']
        if cat_id not in cv_dict[f_name]:
            cv_dict[f_name].append(cat_id)
    return cv_dict

def cluster_vector(cat_vn, class_num):
    """
    Prepare cluster vector, labels must some to one
        cat_v: dict {fn: [labels]}
        class_num: number of classes (90 for mscoco)
    """
    cv_dict = {}
    for key in cat_vn:
        zv = np.zeros(class_num + 1)
        labels = cat_vn[key]
        zv[labels] = 1
        c_v = zv / zv.sum()
        cv_dict[key] = c_v
    return cv_dict

# read json annotation files
with open(TRAIN_VAL) as rf:
    train = json.load(rf)

with open(VAL_ANN) as rf:
    val = json.load(rf)

train.keys()

# mscoco unused ids in range 0-90(inclusive)
range_max = set(range(91))
cats = set()
for entry in train['categories']:
    cats.add(entry['id'])
unused_cats = range_max.difference(cats)
print(unused_cats)

# training cluster vector
train_ifn = imid_fn(train['images'])
train_cv = cat_vector(train['annotations'], train_ifn)
train_cv = cluster_vector(train_cv, 90)

# validation cluster vector
val_ifn = imid_fn(val['images'])
val_cv = cat_vector(val['annotations'], val_ifn)
val_cv = cluster_vector(val_cv, 90)

# test, see, that dicts dont include all images from caption set
print(len(list(train_cv.keys())))
print(len(list(val_cv.keys())))

# concateate 2 dictionaries, more convenient
c_v = dict(train_cv, **val_cv)

# serialize
if not os.path.exists('./obj_vectors'):
    os.makedirs('./obj_vectors')
with open('./obj_vectors/c_v.pickle', 'wb') as wf:
    pickle.dump(c_v, wf)

