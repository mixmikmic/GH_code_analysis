# https://ipython.org/ipython-doc/3/config/extensions/autoreload.html
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os, sys
import torch
import torchvision

import numpy as np
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')

os.environ['CIFAR10_ROOT'] = '/media/user/fast_storage/tensorpack_data/cifar10_data/'
CIFAR10_ROOT = os.environ['CIFAR10_ROOT']

sys.path.append("..")

from torchvision.transforms import Compose, Normalize, ToTensor, Lambda
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from common_utils.dataflow import TransformedDataset, OnGPUDataLoader
from common_utils.imgaug import ToNumpy, RandomOrder, RandomChoice, RandomFlip, RandomAffine, ColorJitter

# Raw datasets: training and testing
train_ds = torchvision.datasets.CIFAR10(root=CIFAR10_ROOT, train=True, download=False)
test_ds = torchvision.datasets.CIFAR10(root=CIFAR10_ROOT, train=False, download=False)

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

resized_train_ds = ResizeDataset(train_ds, output_size=(42, 42))
resized_test_ds = ResizeDataset(test_ds, output_size=(42, 42))

from sklearn.model_selection import StratifiedKFold

n_samples = len(resized_train_ds)
X = np.zeros(n_samples)
Y = np.zeros(n_samples)
for i, (_, label) in enumerate(resized_train_ds):
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

# Data augmentations :
# following pytorch.org/docs/master/torchvision/models.html
mean_val = [0.485, 0.456, 0.406]
std_val = [0.229, 0.224, 0.225]

# Setup data augmentations
train_transforms = Compose([
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
    ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    Normalize(mean_val, std_val)
])
  

test_transforms = Compose([
    ToNumpy(),    
    # Geometry
    RandomChoice([
        RandomAffine(rotation=(-60, 60), scale=(0.95, 1.05), translate=(0.05, 0.05)),
        RandomFlip(proba=0.5, mode='h'),
        RandomFlip(proba=0.5, mode='v'),        
    ]),        
    # To Tensor (float, CxHxW, [0.0, 1.0])  + Normalize
    ToTensor(),
    # Color
    ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    Normalize(mean_val, std_val)
])

data_aug_train_ds = TransformedDataset(resized_train_ds, x_transforms=train_transforms)
data_aug_val_ds = TransformedDataset(resized_train_ds, x_transforms=test_transforms)
data_aug_test_ds = TransformedDataset(resized_test_ds, x_transforms=test_transforms)

split_index = 0 

_DataLoader = OnGPUDataLoader if torch.cuda.is_available() else DataLoader
    
    
train_batches_ds = _DataLoader(data_aug_train_ds, 
                               batch_size=64, 
                               sampler=kfold_samplers[split_index]["train"], 
                               num_workers=0, 
                               drop_last=True, 
                               pin_memory=False)

val_batches_ds = _DataLoader(data_aug_val_ds, 
                             batch_size=64, 
                             sampler=kfold_samplers[split_index]["val"], 
                             num_workers=0, 
                             drop_last=True, 
                             pin_memory=False)

test_batches_ds = _DataLoader(data_aug_test_ds, 
                              batch_size=64, 
                              num_workers=0, 
                              drop_last=True, 
                              pin_memory=False)

import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

from tqdm import tqdm

n_classes = 10
mean_n_classes = []
std_n_classes = []
cnt = 0

classes_stats_per_batches = np.zeros((len(train_batches_ds), n_classes), dtype=np.int)

pbar = tqdm(total=len(train_batches_ds))
for i, (batch_x, batch_y) in enumerate(train_batches_ds):
    for y in batch_y:
        classes_stats_per_batches[i, y] += 1

    postfix_str = "c0: %i | c1: %i" % (classes_stats_per_batches[i, 0], classes_stats_per_batches[i, 1])
    pbar.set_postfix_str(postfix_str, refresh=True)
    pbar.update(1)    
    
pbar.close()

sns.boxplot(data=classes_stats_per_batches, palette="PRGn")

mean_n_classes = np.mean(np.cumsum(classes_stats_per_batches, axis=0), axis=1)
std_n_classes = np.std(np.cumsum(classes_stats_per_batches, axis=0), axis=1)

plt.plot(mean_n_classes)
plt.plot(mean_n_classes + 3.0 * std_n_classes)
plt.plot(mean_n_classes - 3.0 * std_n_classes)



from torchvision.models import SqueezeNet

from common_utils.nn_utils import print_trainable_parameters

squeezenet = SqueezeNet(num_classes=10, version=1.1)
print(squeezenet)

print_trainable_parameters(squeezenet)

if torch.cuda.is_available():
    squeezenet = squeezenet.cuda()

from torch.nn import AvgPool2d, Sequential, MaxPool2d, Conv2d

squeezenet.features[0], squeezenet.features[2] 

from torch.autograd import Variable

test_random_x = Variable(torch.randn(1, 3, 48, 48))
if torch.cuda.is_available():
    test_random_x.data = test_random_x.data.pin_memory()
    test_random_x = test_random_x.cuda(async=True)
    
squeezenet.eval()
test_output_y0 = squeezenet.features[0](test_random_x)
test_output_y1 = Sequential(squeezenet.features[0], squeezenet.features[1], squeezenet.features[2])(test_random_x)
test_output_y0.size(), test_output_y1.size()

layers = [l for i, l in enumerate(squeezenet.features) if i != 2]
layers[0] = Conv2d(3, 64, kernel_size=(3, 3), padding=1)

squeezenet.features = Sequential(*layers)

print(squeezenet)

if torch.cuda.is_available():
    squeezenet = squeezenet.cuda()

from torch.autograd import Variable

test_random_x = Variable(torch.randn(1, 3, 42, 42))
if torch.cuda.is_available():
    test_random_x.data = test_random_x.data.pin_memory()
    test_random_x = test_random_x.cuda(async=True)

squeezenet.eval()
test_output_y = squeezenet.features(test_random_x)
test_output_y.size()

from torch.nn import AvgPool2d, Sequential

layers = [l for l in squeezenet.classifier]
layers[-1] = AvgPool2d(10)

squeezenet.classifier = Sequential(*layers)

print(squeezenet)

from torch.autograd import Variable

test_random_x = Variable(torch.randn(1, 3, 48, 48))
if torch.cuda.is_available():
    test_random_x.data = test_random_x.data.pin_memory()
    test_random_x = test_random_x.cuda(async=True)

squeezenet.eval()
test_output_y = squeezenet(test_random_x)
test_output_y



from torch.nn import CrossEntropyLoss
from torch.optim import Adam

# define loss function (criterion) and optimizer
criterion = CrossEntropyLoss()
if torch.cuda.is_available():
    criterion = CrossEntropyLoss().cuda()
optimizer = Adam(squeezenet.parameters())

squeezenet.eval()

for i, (batch_x, batch_y) in enumerate(train_batches_ds):
    
    batch_x = Variable(batch_x, requires_grad=True)
    batch_y = Variable(batch_y)
    batch_y_pred = squeezenet(batch_x)
    print(type(batch_y.data), type(batch_y_pred.data), batch_y.size(), batch_y_pred.size())
    loss = criterion(batch_y_pred, batch_y)
    print("Loss : ", loss.data)
    break



from common_utils.training_utils import AverageMeter, accuracy, save_checkpoint
from common_utils.training_utils import train_one_epoch, validate

get_ipython().run_line_magic('pinfo2', 'train_one_epoch')

get_ipython().run_line_magic('pinfo2', 'validate')

from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
scheduler = ExponentialLR(optimizer, gamma=0.95)

start_epoch = 0 
n_epochs = 10
model = squeezenet
init_lr = 0.0001

from torch.backends import cudnn
cudnn.benchmark

cudnn.benchmark = True

optimizer = Adam(squeezenet.parameters(), lr=init_lr)

from datetime import datetime


best_acc = 0
now = datetime.now()

if not os.path.exists('logs'):
    os.makedirs('logs')
    
logs_path = os.path.join('logs', 'cifar10_squeezenet_%s' % now.strftime("%Y%m%d_%H%M"))
if not os.path.exists(logs_path):
    os.makedirs(logs_path)    

    
for epoch in range(start_epoch, n_epochs):
    
    scheduler.step()

    # train for one epoch
    ret = train_one_epoch(model, train_batches_ds, criterion, optimizer, epoch, n_epochs, avg_metrics=[accuracy, ])
    if ret is None:
        break
    loss, acc = ret

    # evaluate on validation set
    ret = validate(model, val_batches_ds, criterion, avg_metrics=[accuracy, ])
    if ret is None:
        break
    val_loss, val_acc = ret

    # remember best accuracy and save checkpoint
    if val_acc > best_acc:
        best_acc = max(val_acc, best_acc)
        save_checkpoint(logs_path, 'val_acc', {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'val_acc': val_acc,
            'optimizer' : optimizer.state_dict()})



from common_utils.training_utils import load_checkpoint

from glob import glob

saved_model_filenames = glob(os.path.join(logs_path, "model_val_acc=*"))
assert len(saved_model_filenames) > 0

saved_model_filename = saved_model_filenames[0]

load_checkpoint(saved_model_filename, squeezenet)

# evaluate on validation set
test_loss, test_acc = validate(squeezenet, test_batches_ds, criterion, avg_metrics=[accuracy, ])
test_acc





