import torch
from PIL import Image
import cv2
import time
import numpy as np
import torchvision.models as models
from torch.optim import lr_scheduler
import torchvision
import torch.nn as nn
from torchvision import datasets, models, transforms,utils
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.autograd import Variable
import os
import pandas as pd
import glob
from skimage import io
import matplotlib.pyplot as plt
import os 

get_ipython().magic('matplotlib inline')
use_gpu = torch.cuda.is_available()

# Config
DATA_DIR = 'data'
TRAIN_DIR = 'train'
TEST_DIR = 'test'
VAL_DIR = 'val'
TRAIN_DS_LOC = os.path.join(DATA_DIR, TRAIN_DIR)
TEST_DS_LOC = os.path.join(DATA_DIR, TEST_DIR)
VAL_DS_LOC = os.path.join(DATA_DIR, VAL_DIR)
IMG_PATTERN = '*.png'
IMG_PROPERTIES_FILE = 'Data_Entry_2017.csv'
LABEL_SEP = '|'
Height=512
Width=512


# Retrieve and process labels
l2i = {'Atelectasis': 0,
 'Cardiomegaly': 1,
 'Consolidation': 2,
 'Edema': 3,
 'Effusion': 4,
 'Emphysema': 5,
 'Fibrosis': 6,
 'Hernia': 7,
 'Infiltration': 8,
 'Mass': 9,
 'No Finding': 10,
 'Nodule': 11,
 'Pleural_Thickening': 12,
 'Pneumonia': 13,
 'Pneumothorax': 14}

properties = pd.read_csv(IMG_PROPERTIES_FILE, skiprows=1, header=None, low_memory=False, na_filter=False).values
labels = {prop[0] : [ l2i[label] for label in  prop[1].split(LABEL_SEP)] for prop in properties}

num_classes=len(l2i)

# Data transforms
data_transforms = {
    'train': transforms.Compose([
          transforms.ToPILImage(),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),

    ]),
}

def label_standard_array(labels):
    standard_class=np.zeros(shape=num_classes)
    for i in labels:
       standard_class[i]=1
    return standard_class
    

#def split TODO:training and test validations
class XrayFolderDS(Dataset):
    def __init__(self, root, transform = None):
        self.data = glob.glob(os.path.join(root, IMG_PATTERN))
        self.transform = transform
        # transforms.ToPILImage(),
        self.to_tensor = transforms.Compose([transforms.ToTensor()])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        entry = self.data[index]
        name = os.path.split(entry)[-1]
        img = cv2.imread(entry)
        # img = Image.open(entry)
        img = self.transform(img)
        labels_name=label_standard_array(labels[name])
        return img,labels_name
        
test_d = XrayFolderDS(TRAIN_DS_LOC,data_transforms['train'])[0][0]
# convert to np and remove extra dim
numpy_d = test_d.numpy()[0] 
print(test_d.size())
plt.imshow(numpy_d, cmap='gray') # Remove gray for green normalized image :D
plt.show()

#Test the loading code
class XrayFolderDS2(Dataset):
    def __init__(self, root, transform = None):
        self.data = glob.glob(os.path.join(root, IMG_PATTERN))
        self.transform = transform
        # transforms.ToPILImage(),
        self.to_tensor = transforms.Compose([transforms.ToTensor()])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        entry = self.data[index]
        name = os.path.split(entry)[-1]
        img = cv2.imread(entry)
        # img = Image.open(entry)
        img = self.transform(img)
        labels_name=label_standard_array(labels[name])
        #labels_name = labels[name]
        return img,labels_name
    
import line_profiler

def ftst():
    
    ds = DataLoader(tds, batch_size=25, shuffle=True)
    s = 0
    t = time.time()
    for e, i in enumerate(ds):
        s+=len(i)
        if e == 1000:
            print(i[0].size())
            break
    print('took {:.4f} seconds '.format(time.time() - t,))

tds = XrayFolderDS2(TRAIN_DS_LOC,data_transforms['train'])

lpx = line_profiler.LineProfiler(tds.__getitem__)
#print(dir(lpx))
lpx(ftst)()
lpx.print_stats()
    

# Hyperparameters
BATCH_SIZE = 25

# Prepare Data
# print( [d for d in os.listdir(TRAIN_DS_LOC) ] )

#print(os.getcwd()+'\\data\\train\\', torchvision.datasets.folder.find_classes('data/train'))
train_dataset = XrayFolderDS(TRAIN_DS_LOC,data_transforms['train'])
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = XrayFolderDS(VAL_DS_LOC,data_transforms['val'])
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

dataloaders={'train':train_dataloader,'val':val_dataloader}
dataset_sizes = {'train':len(train_dataset),'val':len(val_dataset)}

# test_dataset = XrayFolderDS(TRAIN_DS_LOC,data_transforms['test'])
# test_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# TODO: Implement k fold cross validation

# def show_xray_batch(sample_batched):
#     """Show image with landmarks for a batch of samples."""
#     xrays_batch,labels_batch = sample_batched
#     print('labels_batch batch size',labels_batch.shape)
#     print(len(labels_batch))
#     grid_1 = utils.make_grid(xrays_batch)
#     plt.imshow(grid_1.numpy().transpose(( 1,2, 0)))


# for i_batch, sample_batched in enumerate(train_dataloader):
#     print ('Batch Number', i_batch)
#     # observe 4th batch and stop.
#     if i_batch == 5:
#         plt.figure()
#         show_xray_batch(sample_batched)
#         plt.axis('off')
#         plt.ioff()
#         plt.show()
#         break

xray_model = models.resnet50(pretrained=True)

num_ftrs = xray_model.fc.in_features
# changing the final_layer for our images 
xray_model.fc = nn.Linear(num_ftrs,num_classes)
print(xray_model.fc)
print(xray_model)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    i = 0 
    start_time = time.time()


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)


        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            print('Total Batches: ', len(dataloaders[phase]))

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                # wrap them in Variable
                
                if phase == 'train':
                    inputs, labels = Variable(inputs), Variable(labels)
                else:
                    inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)
                
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
               

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = torch.sigmoid(model(inputs))
                preds = torch.sign(torch.sign(outputs.data-0.5)+1)
                # print(outputs.size())
                # print(labels)
                loss = criterion(outputs.double(), labels)
                
                if i%100==0:
                    print("i is",i)
                    print("time passed. {:.2f} ".format(time.time()-start_time))
                    print(' Loss: {:.4f} '.format(loss.data[0]))
                    
        
                i=i+1

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                part_corr = torch.sum((preds == labels.data.float()),1).double()
                # Average fraction of correct labels
                corr_avg =  torch.mean(part_corr)/float(num_classes)
                #corrects = torch.sum(part_corr == num_classes)
                running_corrects += torch.sum(part_corr)
                
                if i%100==0:
                    print('Batch Accuracy ', corr_avg)

            epoch_loss = running_loss / float(dataset_sizes[phase])
            epoch_acc = running_corrects / (float(dataset_sizes[phase]) * float(num_classes))


            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc ))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
            
            if phase == 'train':
                torch.save(model.state_dict(), open("projectx_%s_%f.model" % (epoch, epoch_loss), 'wb'))



    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if use_gpu:
    xray_model = xray_model.cuda()
    print("GPU POWER FTW")

criterion = nn.MultiLabelSoftMarginLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(xray_model.parameters(), lr=0.01, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(xray_model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=5)



