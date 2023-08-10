# https://ipython.org/ipython-doc/3/config/extensions/autoreload.html
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os, sys
import numpy as np
import cv2

sys.path.append("..")

HAS_GPU = True

import torch
from torch.autograd import Variable
from torch.nn.functional import sigmoid

from model import SiameseNetworks
from common_utils.training_utils import load_checkpoint
from glob import glob

logs_path = os.path.join('logs', 'siamese_networks_verification_task_20171125_2242')

siamese_net = SiameseNetworks(input_shape=(105, 105, 1))
if HAS_GPU and torch.cuda.is_available():
    siamese_net = siamese_net.cuda()
    
best_model_filenames = glob(os.path.join(logs_path, "model_val_acc=*"))
assert len(best_model_filenames) == 1
load_checkpoint(best_model_filenames[0], siamese_net)

np.random.seed(12345)

OMNIGLOT_REPO_PATH='omniglot'

TEST_DATA_PATH = os.path.join(OMNIGLOT_REPO_PATH, 'python', 'images_evaluation')
test_alphabets = get_ipython().getoutput('ls {TEST_DATA_PATH}')
test_alphabets = list(test_alphabets)

assert len(test_alphabets) > 1, "%s" % test_alphabets[0]        
        
test_alphabet_char_id_drawer_ids = {}
for a in test_alphabets:
    res = get_ipython().getoutput('ls "{os.path.join(TEST_DATA_PATH, a)}"')
    char_ids = list(res)
    test_alphabet_char_id_drawer_ids[a] = {}
    for char_id in char_ids:
        res = get_ipython().getoutput('ls "{os.path.join(TEST_DATA_PATH, a, char_id)}"')
        test_alphabet_char_id_drawer_ids[a][char_id] = [_id[:-4] for _id in list(res)]


# Sample 12 drawers out of 20
all_drawers_ids = np.arange(20) 
train_drawers_ids = np.random.choice(all_drawers_ids, size=12, replace=False)
# Sample 4 drawers out of remaining 8
val_drawers_ids = np.random.choice(list(set(all_drawers_ids) - set(train_drawers_ids)), size=4, replace=False)
test_drawers_ids = np.array(list(set(all_drawers_ids) - set(val_drawers_ids) - set(train_drawers_ids)))

def create_str_drawers_ids(drawers_ids):
    return ["_{0:0>2}".format(_id) for _id in drawers_ids]

val_drawers_ids = create_str_drawers_ids(val_drawers_ids)
test_drawers_ids = create_str_drawers_ids(test_drawers_ids)

from torchvision.transforms import ToTensor
from common_utils.dataflow import TransformedDataset

from common_utils.dataflow_visu_utils import display_basic_dataset
from dataflow import OmniglotDataset

import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')

random_alphabet_index = np.random.randint(len(test_alphabet_char_id_drawer_ids))
alphabet = list(test_alphabet_char_id_drawer_ids.keys())[random_alphabet_index]
alphabet

two_test_drawers_ids = np.random.choice(test_drawers_ids, size=2, replace=False)

alphabet_char_id_drawers_ids = {}
char_id_drawer_ids = test_alphabet_char_id_drawer_ids[alphabet]    
random_chars = np.random.choice(list(char_id_drawer_ids.keys()), size=20, replace=False).tolist()
alphabet_char_id_drawers_ids[alphabet] = {}
for char_id in char_id_drawer_ids:
    if char_id in random_chars:
        alphabet_char_id_drawers_ids[alphabet][char_id] = char_id_drawer_ids[char_id]
        
test_images_ds = OmniglotDataset("Test", data_path=TEST_DATA_PATH, 
                          alphabet_char_id_drawers_ids=alphabet_char_id_drawers_ids, 
                          drawers_ids=two_test_drawers_ids[0])

additional_images_ds = OmniglotDataset("Test", data_path=TEST_DATA_PATH, 
                          alphabet_char_id_drawers_ids=alphabet_char_id_drawers_ids, 
                          drawers_ids=two_test_drawers_ids[1])    
random_chars

len(test_images_ds), len(additional_images_ds)

# Transform string label to class index:
y_transform = lambda y: torch.FloatTensor([random_chars.index(os.path.basename(y))])

test_images = TransformedDataset(test_images_ds, x_transforms=ToTensor(), y_transforms=y_transform)
additional_images = TransformedDataset(additional_images_ds, x_transforms=ToTensor(), y_transforms=y_transform)

batch_additional_x = []
batch_additional_y = []

for x, y in additional_images:
    batch_additional_x.append(x)
    batch_additional_y.append(y)

batch_additional_x = torch.cat(batch_additional_x).unsqueeze(dim=1)
batch_additional_y = torch.cat(batch_additional_y)

if HAS_GPU and torch.cuda.is_available():
    batch_additional_x = batch_additional_x.cuda()
    
batch_additional_x = Variable(batch_additional_x, volatile=True)

from common_utils.dataflow_visu_utils import _to_ndarray

siamese_net.eval()

accuracy = 0.0
accuracy_top3 = 0.0

for test_x, test_y in test_images:
    batch_test_x = test_x.expand_as(batch_additional_x)
    if HAS_GPU and torch.cuda.is_available():
        batch_test_x = batch_test_x.cuda()
        
    batch_test_x = Variable(batch_test_x, volatile=True)
    
    y_logits = siamese_net(batch_test_x, batch_additional_x)
    y_probas = sigmoid(y_logits)
    y_probas_topk, indices_topk = torch.topk(y_probas.data, k=3, dim=0, largest=True)
    if indices_topk.is_cuda:
        indices_topk = indices_topk.cpu()
    if len(indices_topk.size()) > 1:
        indices_topk = indices_topk.view(-1)        
    classes_topk = batch_additional_y[indices_topk]
    
    accuracy += (test_y == classes_topk[0]).sum()
    accuracy_top3 += (test_y == classes_topk[:3]).sum()
    
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    index = int(test_y[0])
    plt.title("Test image, %s" % random_chars[index])
    plt.imshow(_to_ndarray(test_x))
    plt.subplot(122)
    index = int(classes_topk[0])
    plt.title("Best similar additional image, %s" % random_chars[index])
    index = int(indices_topk[0])
    plt.imshow(_to_ndarray(batch_additional_x.data[index, :, :, :]))
        
accuracy /= len(test_images)
accuracy_top3 /= len(test_images)

print("One-shot learning accuracy: ", accuracy)
print("One-shot learning accuracy@3: ", accuracy_top3)



siamese_net.eval()
two_test_drawers_ids = np.random.choice(test_drawers_ids, size=2, replace=False)

mean_accuracy = []
mean_accuracy_top3 = []

for alphabet in test_alphabet_char_id_drawer_ids:
    
    print("Alphabet: ", alphabet)
    
    for _ in range(2):
    
        alphabet_char_id_drawers_ids = {}
        char_id_drawer_ids = test_alphabet_char_id_drawer_ids[alphabet]    
        random_chars = np.random.choice(list(char_id_drawer_ids.keys()), size=20, replace=False).tolist()
        alphabet_char_id_drawers_ids[alphabet] = {}
        for char_id in char_id_drawer_ids:
            if char_id in random_chars:
                alphabet_char_id_drawers_ids[alphabet][char_id] = char_id_drawer_ids[char_id]

        test_images_ds = OmniglotDataset("Test", data_path=TEST_DATA_PATH, 
                                  alphabet_char_id_drawers_ids=alphabet_char_id_drawers_ids, 
                                  drawers_ids=two_test_drawers_ids[0])

        additional_images_ds = OmniglotDataset("Test", data_path=TEST_DATA_PATH, 
                                  alphabet_char_id_drawers_ids=alphabet_char_id_drawers_ids, 
                                  drawers_ids=two_test_drawers_ids[1])

        # Transform string label to class index:
        y_transform = lambda y: torch.FloatTensor([random_chars.index(os.path.basename(y))])

        test_images = TransformedDataset(test_images_ds, x_transforms=ToTensor(), y_transforms=y_transform)
        additional_images = TransformedDataset(additional_images_ds, x_transforms=ToTensor(), y_transforms=y_transform)

        batch_additional_x = []
        batch_additional_y = []

        for x, y in additional_images:
            batch_additional_x.append(x)
            batch_additional_y.append(y)

        batch_additional_x = torch.cat(batch_additional_x).unsqueeze(dim=1)
        batch_additional_y = torch.cat(batch_additional_y)

        if HAS_GPU and torch.cuda.is_available():
            batch_additional_x = batch_additional_x.cuda()

        batch_additional_x = Variable(batch_additional_x, volatile=True)

        accuracy = 0.0
        accuracy_top3 = 0.0

        for test_x, test_y in test_images:
            batch_test_x = test_x.expand_as(batch_additional_x)
            if HAS_GPU and torch.cuda.is_available():
                batch_test_x = batch_test_x.cuda()

            batch_test_x = Variable(batch_test_x, volatile=True)

            y_logits = siamese_net(batch_test_x, batch_additional_x)
            y_probas = sigmoid(y_logits)
            y_probas_topk, indices_topk = torch.topk(y_probas.data, k=3, dim=0, largest=True)
            if indices_topk.is_cuda:
                indices_topk = indices_topk.cpu()
            if len(indices_topk.size()) > 1:
                indices_topk = indices_topk.view(-1)        
            classes_topk = batch_additional_y[indices_topk]

            accuracy += (test_y == classes_topk[0]).sum()
            accuracy_top3 += (test_y == classes_topk[:3]).sum()

        accuracy /= len(test_images)
        accuracy_top3 /= len(test_images)

        print("One-shot learning accuracy: ", accuracy)
        print("One-shot learning accuracy@3: ", accuracy_top3)
        mean_accuracy.append(accuracy)
        mean_accuracy_top3.append(accuracy_top3)        
    
mean_accuracy = np.mean(mean_accuracy)
mean_accuracy_top3 = np.mean(mean_accuracy_top3)
print("\nOne-shot learning mean accuracy: ", mean_accuracy)
print("One-shot learning mean accuracy@3: ", mean_accuracy_top3)



