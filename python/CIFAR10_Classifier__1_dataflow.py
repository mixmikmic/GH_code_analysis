# https://ipython.org/ipython-doc/3/config/extensions/autoreload.html
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from __future__ import print_function

import os, sys
import torch
import torchvision

import numpy as np
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')

os.environ['CIFAR10_ROOT'] = '/media/user/fast_storage/tensorpack_data/cifar10_data/'
sys.path.append("..")

CIFAR10_ROOT = os.environ['CIFAR10_ROOT']

train_ds = torchvision.datasets.CIFAR10(root=CIFAR10_ROOT, train=True, download=False)
test_ds = torchvision.datasets.CIFAR10(root=CIFAR10_ROOT, train=False, download=False)

len(train_ds), len(test_ds)

train_dp = train_ds[100]
test_dp = test_ds[100]
type(train_dp), type(train_dp[0]), type(train_dp[1])

plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.imshow(train_dp[0])
plt.title("Train sample : class %i" % train_dp[1])
plt.subplot(122)
plt.imshow(test_dp[0])
_ = plt.title("Test sample : class %i" % test_dp[1])

from common_utils.dataflow_visu_utils import display_basic_dataset, display_data_augmentations, display_batches

display_basic_dataset(train_ds)

display_basic_dataset(test_ds)

from PIL import Image
from torch.utils.data import Dataset

class ResizeDataset(Dataset):
    
    def __init__(self, ds, output_size=(32, 32)):        
        assert isinstance(ds, Dataset)        
        self.ds = ds
        self.output_size = output_size
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        x, y = self.ds[index]
        x = x.resize(self.output_size)
        return x, y
    

resized_train_ds = ResizeDataset(train_ds, output_size=(48, 48))
resized_test_ds = ResizeDataset(test_ds, output_size=(48, 48))

display_basic_dataset(resized_train_ds, max_datapoints=5)

class TransformedDataset(Dataset):
    
    def __init__(self, ds, x_transforms, y_transforms=None):
        assert isinstance(ds, Dataset)
        assert callable(x_transforms)
        if y_transforms is not None:
            assert callable(y_transforms)        
        self.ds = ds        
        self.x_transforms = x_transforms
        self.y_transforms = y_transforms
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        x, y = self.ds[index]       
        x = self.x_transforms(x)
        if self.y_transforms is not None:
            y = self.y_transforms(y)
        return x, y

from torchvision.transforms import Compose, Normalize, ToTensor, Lambda
from torchvision.transforms import RandomCrop

from common_utils.imgaug import ToNumpy, RandomOrder, RandomChoice, RandomFlip, RandomAffine, ColorJitter

mean_val = [0.5] * 3  # RGB
std_val = [0.5] * 3  # RGB

train_transforms = Compose([
    # From 48 -> 42
    RandomCrop(42),
    ToNumpy(),
    # Geometry
    RandomChoice([
        RandomAffine(rotation=(-60, 60), scale=(0.95, 1.05), translate=(0.05, 0.05)),
        RandomFlip(proba=0.5, mode='h'),
        RandomFlip(proba=0.5, mode='v'),        
    ]),    
    # To Tensor (float, CxHxW, [0.0, 1.0]) + Normalize
    ToTensor(),
    # Color
    ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    Normalize(mean_val, std_val)
])
  

test_transforms = Compose([
    RandomCrop(32),
    ToNumpy(),    
    # Geometry
    RandomChoice([
        RandomAffine(rotation=(-60, 60), scale=(0.90, 1.1), translate=(0.15, 0.15)),
        RandomFlip(proba=0.5, mode='h'),
        RandomFlip(proba=0.5, mode='v'),        
    ]),        
    # To Tensor (float, CxHxW, [0.0, 1.0])  + Normalize
    ToTensor(),
    # Color
    ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35),
    Normalize(mean_val, std_val)
])

data_aug_train_ds = TransformedDataset(resized_train_ds, x_transforms=train_transforms)
data_aug_val_ds = TransformedDataset(resized_train_ds, x_transforms=test_transforms)

display_data_augmentations(resized_train_ds, data_aug_train_ds, max_datapoints=10)

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from sklearn.model_selection import StratifiedKFold

ds = resized_train_ds

n_samples = len(ds)
X = np.zeros(n_samples)
Y = np.zeros(n_samples)
for i, (_, label) in enumerate(ds):
    Y[i] = label

kfolds_train_indices = []
kfolds_val_indices = []
    
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
for train_indices, val_indices in skf.split(X, Y):
    kfolds_train_indices.append(train_indices)
    kfolds_val_indices.append(val_indices)    

kfold_samplers = []
for train_indices, val_indices in zip(kfolds_train_indices, kfolds_val_indices):
    kfold_samplers.append({"train": SubsetRandomSampler(train_indices), 
                           "val": SubsetRandomSampler(val_indices)})

# For visualization purposes:
n_samples = 10
kfold_samplers = []
for train_indices, val_indices in zip(kfolds_train_indices, kfolds_val_indices):
    kfold_samplers.append({"train": SubsetRandomSampler(train_indices[:n_samples]), 
                           "val": SubsetRandomSampler(val_indices[:n_samples])})

split_index = 0 

train_batches_ds = DataLoader(data_aug_train_ds, 
                              batch_size=4, 
                              sampler=kfold_samplers[split_index]["train"], num_workers=4, drop_last=True)
val_batches_ds = DataLoader(data_aug_val_ds, 
                            batch_size=4, 
                            sampler=kfold_samplers[split_index]["val"], num_workers=4, drop_last=True)

max_epochs = 5

for epoch in range(max_epochs):     
    display_batches(train_batches_ds, max_batches=-1, figsize=(16, 4), suptitle_prefix="Epoch %i | " % epoch)

kfold_samplers = []
for train_indices, val_indices in zip(kfolds_train_indices, kfolds_val_indices):
    kfold_samplers.append({"train": SubsetRandomSampler(train_indices), 
                           "val": SubsetRandomSampler(val_indices)})

split_index = 0 

train_batches_ds = DataLoader(data_aug_train_ds, 
                              batch_size=64, 
                              sampler=kfold_samplers[split_index]["train"], num_workers=4, drop_last=True)
val_batches_ds = DataLoader(data_aug_val_ds, 
                            batch_size=64, 
                            sampler=kfold_samplers[split_index]["val"], num_workers=4, drop_last=True)

display_batches(train_batches_ds, max_batches=2, n_cols=10)



len(train_batches_ds), len(val_batches_ds)

get_ipython().run_cell_magic('timeit', '-r1 -n3', '\nfor i, (batch_x, batch_y) in enumerate(train_batches_ds):\n    pass')

get_ipython().run_cell_magic('timeit', '-r1 -n3', '\nfor i, (batch_x, batch_y) in enumerate(val_batches_ds):\n    pass')



from collections import defaultdict, Hashable
total_y_stats = defaultdict(int)
cnt = 0
display_freq = 10
display_total = True

for i, (batch_x, batch_y) in enumerate(train_batches_ds):
    y_stats = defaultdict(int)
    for y in batch_y:
        if isinstance(y, Hashable):
            total_y_stats[y] += 1 
            y_stats[y] += 1
                    
    if (cnt % display_freq) == 0:
        print("\n%i | Labels counts: " % cnt)
        print("  current: | ", end='')
        for k in y_stats:
            print("'{}': {} |".format(str(k), y_stats[k]), end=' ')
        print('')
        if display_total:
            print("    total: | ", end='')
            for k in total_y_stats:
                print("'{}': {} |".format(str(k), total_y_stats[k]), end=' ')
            print('')                    
    cnt += 1



torch.cuda.is_available()

get_ipython().system('nvidia-smi --format=csv --query-gpu=memory.used')

t = torch.randn((32, 3, 512, 512)).cuda()
p = torch.randn((32, 3, 512, 512))

get_ipython().system('nvidia-smi --format=csv --query-gpu=memory.used')

t.is_cuda, p.is_cuda

pp = p.pin_memory()

p.is_pinned(), pp.is_pinned(), pp.is_cuda

ppc = pp.cuda(async=True)
ppc.is_pinned(), ppc.is_cuda

get_ipython().system('nvidia-smi --format=csv --query-gpu=memory.used')



from common_utils.dataflow import OnGPUDataLoader

split_index = 0 

cuda_train_batches_ds = OnGPUDataLoader(data_aug_train_ds, 
                                        batch_size=64, 
                                        sampler=kfold_samplers[split_index]["train"], 
                                        num_workers=4, 
                                        drop_last=True, 
                                        pin_memory=True)
cuda_val_batches_ds = OnGPUDataLoader(data_aug_val_ds,
                                      batch_size=64, 
                                      sampler=kfold_samplers[split_index]["val"],
                                      num_workers=4, 
                                      drop_last=True)

for batch_x, batch_y in cuda_train_batches_ds:
    print(batch_x.size(), batch_y.size())
    print(type(batch_x), type(batch_y))    
    break

get_ipython().system('nvidia-smi --format=csv --query-gpu=memory.used')

for i, (batch_x, batch_y) in enumerate(cuda_train_batches_ds):
    print(batch_x.is_cuda, batch_y.is_cuda)
    break
    pass

get_ipython().system('nvidia-smi --format=csv --query-gpu=memory.used')

display_batches(cuda_val_batches_ds, max_batches=2, n_cols=10)



get_ipython().run_cell_magic('timeit', '-r1 -n3', '\nfor i, (batch_x, batch_y) in enumerate(cuda_train_batches_ds):\n    s = torch.sum(batch_x, dim=1)')





