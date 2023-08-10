# https://ipython.org/ipython-doc/3/config/extensions/autoreload.html
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os, sys
import numpy as np
import cv2

sys.path.append("..")

OMNIGLOT_REPO_PATH = "omniglot"

# !unzip --help 

if OMNIGLOT_REPO_PATH is None or not os.path.exists(OMNIGLOT_REPO_PATH):
    OMNIGLOT_REPO_PATH = "omniglot"
    get_ipython().system('git clone https://github.com/brendenlake/omniglot')
    get_ipython().system('cd {os.path.join(OMNIGLOT_REPO_PATH, "python")} && unzip images_background.zip        ')
    get_ipython().system('cd {os.path.join(OMNIGLOT_REPO_PATH, "python")} && unzip images_evaluation.zip')

TRAIN_DATA_PATH = os.path.join(OMNIGLOT_REPO_PATH, 'python', 'images_background')
train_alphabets = get_ipython().getoutput('ls {TRAIN_DATA_PATH}')
train_alphabets = list(train_alphabets)
print("\nTrain alphabets: \n", train_alphabets, len(train_alphabets))

TEST_DATA_PATH = os.path.join(OMNIGLOT_REPO_PATH, 'python', 'images_evaluation')
test_alphabets = get_ipython().getoutput('ls {TEST_DATA_PATH}')
test_alphabets = list(test_alphabets)
print("\nEvaluation alphabets: \n", test_alphabets, len(test_alphabets))

train_alphabet_char_id_drawer_ids = {}
for a in train_alphabets:
    res = get_ipython().getoutput('ls "{os.path.join(TRAIN_DATA_PATH, a)}"')
    char_ids = list(res)
    train_alphabet_char_id_drawer_ids[a] = {}
    for char_id in char_ids:
        res = get_ipython().getoutput('ls "{os.path.join(TRAIN_DATA_PATH, a, char_id)}"')
        train_alphabet_char_id_drawer_ids[a][char_id] = [_id[:-4] for _id in list(res)]

test_alphabet_char_id_drawer_ids = {}
for a in test_alphabets:
    res = get_ipython().getoutput('ls "{os.path.join(TEST_DATA_PATH, a)}"')
    char_ids = list(res)
    test_alphabet_char_id_drawer_ids[a] = {}
    for char_id in char_ids:
        res = get_ipython().getoutput('ls "{os.path.join(TEST_DATA_PATH, a, char_id)}"')
        test_alphabet_char_id_drawer_ids[a][char_id] = [_id[:-4] for _id in list(res)]

print("Characters of 'Alphabet_of_the_Magi': \n", train_alphabet_char_id_drawer_ids['Alphabet_of_the_Magi'].keys())

print("Images of a single character of 'Alphabet_of_the_Magi': \n", 
      train_alphabet_char_id_drawer_ids['Alphabet_of_the_Magi']['character01'])



def get_image(group, char_id, _id, _type="Train"):    
    assert _type in ["Train", "Test"]
    path = TRAIN_DATA_PATH if _type == "Train" else TEST_DATA_PATH
    path = os.path.join(path, group, char_id, "%s.png" % _id)
    assert os.path.exists(path), "Path '%s' does not exist" % path
    img = cv2.imread(path)
    return img

import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(12, 4))
plt.suptitle("Training")
plt.subplot(131)
plt.imshow(get_image('Balinese', 'character01', '0108_01'))
plt.subplot(132)
plt.imshow(get_image('Balinese', 'character01', '0108_02'))
plt.subplot(133)
plt.imshow(get_image('Balinese', 'character01', '0108_03'))

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(get_image('Balinese', 'character02', '0109_01'))
plt.subplot(132)
plt.imshow(get_image('Balinese', 'character02', '0109_02'))
plt.subplot(133)
_ = plt.imshow(get_image('Balinese', 'character02', '0109_03'))

plt.figure(figsize=(12, 4))
plt.suptitle("Testing")
plt.subplot(131)
plt.imshow(get_image('Angelic', 'character01', '0965_01', "Test"))
plt.subplot(132)
plt.imshow(get_image('Angelic', 'character01', '0965_02', "Test"))
plt.subplot(133)
plt.imshow(get_image('Angelic', 'character01', '0965_03', "Test"))

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(get_image('Angelic', 'character02', '0966_01', "Test"))
plt.subplot(132)
plt.imshow(get_image('Angelic', 'character02', '0966_02', "Test"))
plt.subplot(133)
_ = plt.imshow(get_image('Angelic', 'character02', '0966_03', "Test"))



from dataflow import OmniglotDataset
from common_utils.dataflow import ResizedDataset
from common_utils.dataflow_visu_utils import display_basic_dataset

train_ds = OmniglotDataset("Train", data_path=TRAIN_DATA_PATH, 
                           alphabet_char_id_drawers_ids=train_alphabet_char_id_drawer_ids)

valtest_ds = OmniglotDataset("Test", data_path=TEST_DATA_PATH, 
                             alphabet_char_id_drawers_ids=test_alphabet_char_id_drawer_ids)

train_ds = ResizedDataset(train_ds, output_size=(80, 80))
valtest_ds = ResizedDataset(valtest_ds, output_size=(80, 80))

display_basic_dataset(train_ds, max_datapoints=50, n_cols=8, figsize=(18, 4))

display_basic_dataset(valtest_ds, max_datapoints=50, n_cols=8, figsize=(18, 4))

np.random.seed(12345)

# Sample 12 drawers out of 20
all_drawers_ids = np.arange(20) 
train_drawers_ids = np.random.choice(all_drawers_ids, size=12, replace=False)
# Sample 4 drawers out of remaining 8
val_drawers_ids = np.random.choice(list(set(all_drawers_ids) - set(train_drawers_ids)), size=4, replace=False)
test_drawers_ids = np.array(list(set(all_drawers_ids) - set(val_drawers_ids) - set(train_drawers_ids)))

def create_str_drawers_ids(drawers_ids):
    return ["_{0:0>2}".format(_id) for _id in drawers_ids]

train_drawers_ids = create_str_drawers_ids(train_drawers_ids)
val_drawers_ids = create_str_drawers_ids(val_drawers_ids)
test_drawers_ids = create_str_drawers_ids(test_drawers_ids)

print(train_drawers_ids)
print(val_drawers_ids)
print(test_drawers_ids)

train_ds = OmniglotDataset("Train", data_path=TRAIN_DATA_PATH, 
                           alphabet_char_id_drawers_ids=train_alphabet_char_id_drawer_ids, 
                           drawers_ids=train_drawers_ids)

val_ds = OmniglotDataset("Test", data_path=TEST_DATA_PATH, 
                         alphabet_char_id_drawers_ids=test_alphabet_char_id_drawer_ids, 
                         drawers_ids=val_drawers_ids)

test_ds = OmniglotDataset("Test", data_path=TEST_DATA_PATH, 
                          alphabet_char_id_drawers_ids=test_alphabet_char_id_drawer_ids, 
                          drawers_ids=test_drawers_ids)

train_ds = ResizedDataset(train_ds, output_size=(80, 80))
val_ds = ResizedDataset(val_ds, output_size=(80, 80))
test_ds = ResizedDataset(test_ds, output_size=(80, 80))

len(train_ds), len(val_ds), len(test_ds)

y_labels = set()
for x, y in train_ds:
    y_labels.add(y)
    
# Number of classes
print("Number of classes in train:", len(y_labels), x.shape)

# number of alphabets:
res = set([y_label.split('/')[0] for y_label in y_labels])
print("Number of alphabets: ", len(res), res)

from dataflow import SameOrDifferentPairsDataset
from common_utils.dataflow_visu_utils import _to_ndarray

train_pairs = SameOrDifferentPairsDataset(train_ds, nb_pairs=int(30e3))
val_pairs = SameOrDifferentPairsDataset(val_ds, nb_pairs=int(10e3))
test_pairs = SameOrDifferentPairsDataset(test_ds, nb_pairs=int(10e3))

len(train_pairs), len(val_pairs), len(test_pairs)

max_datapoints = 10
n_cols = 5
for i, ((x1, x2), y) in enumerate(train_pairs):

    if i % n_cols == 0:
        plt.figure(figsize=(12, 4))

    x1 = _to_ndarray(x1)
    x2 = _to_ndarray(x2)    
    plt.subplot(2, n_cols, (i % n_cols) + 1)
    plt.imshow(x1)
    plt.title("Class %i" % y)
    plt.subplot(2, n_cols, (i % n_cols) + 1 + n_cols)
    plt.imshow(x2)

    max_datapoints -= 1
    if max_datapoints == 0:
        break

from common_utils.imgaug import RandomAffine, RandomApply
from common_utils.dataflow import TransformedDataset
from torchvision.transforms import Compose, ToTensor, Normalize
from dataflow import PairTransformedDataset

train_data_aug = Compose([
    RandomApply(
        RandomAffine(rotation=(-10, 10), scale=(0.8, 1.2), translate=(-0.05, 0.05)),
        proba=0.5
    ),
    ToTensor()
])

test_data_aug = Compose([
    ToTensor()
])

train_aug_pairs = PairTransformedDataset(train_pairs, x_transforms=train_data_aug)
val_aug_pairs = PairTransformedDataset(val_pairs, x_transforms=test_data_aug)
test_aug_pairs = PairTransformedDataset(test_pairs, x_transforms=test_data_aug)

max_datapoints = 10
n_cols = 5
for i, ((x1, x2), y) in enumerate(train_aug_pairs):

    if i % n_cols == 0:
        plt.figure(figsize=(12, 4))

    x1 = _to_ndarray(x1)
    x2 = _to_ndarray(x2)    
    plt.subplot(2, n_cols, (i % n_cols) + 1)
    plt.imshow(x1)
    plt.title("Class %i" % y)
    plt.subplot(2, n_cols, (i % n_cols) + 1 + n_cols)
    plt.imshow(x2)

    max_datapoints -= 1
    if max_datapoints == 0:
        break

from torch.utils.data import DataLoader

batch_size = 5
train_batches = DataLoader(train_aug_pairs, batch_size=batch_size, 
                           shuffle=True, num_workers=1, 
                           drop_last=True)

val_batches = DataLoader(val_aug_pairs, batch_size=batch_size, 
                         shuffle=True, num_workers=1,
                         drop_last=True)

test_batches = DataLoader(test_aug_pairs, batch_size=batch_size, 
                          shuffle=False, num_workers=1,                   
                          drop_last=False)


len(train_batches), len(val_batches), len(test_batches)

max_datapoints = 5
n_cols = 5
for i, ((batch_x1, batch_x2), batch_y) in enumerate(train_batches):

    print(batch_x1.size(), batch_x2.size(), batch_y.size())
    
    plt.figure(figsize=(16, 4))
    plt.suptitle("Batch %i" % i)
    for j in range(len(batch_x1)):
        if j > 0 and j % n_cols == 0:
            plt.figure(figsize=(16, 4))
        
        x1 = batch_x1[j, ...]
        x2 = batch_x2[j, ...]
        y = batch_y[j, ...]
    
        x1 = _to_ndarray(x1)
        x2 = _to_ndarray(x2)    
        plt.subplot(2, n_cols, (j % n_cols) + 1)
        plt.imshow(x1)
        plt.title("Class %i" % y)
        plt.subplot(2, n_cols, (j % n_cols) + 1 + n_cols)
        plt.imshow(x2)

    max_datapoints -= 1
    if max_datapoints == 0:
        break





