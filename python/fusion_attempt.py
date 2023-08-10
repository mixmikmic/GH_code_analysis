get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import os
import sys

from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

from torchvision import datasets, models, transforms

# Base Directory where data is stored
PATH = '/media/rene/Data/camelyon_out/tiles_299_100t/'

fast_ai_dir = '/media/rene/Data/fastai/'
sys.path.append(fast_ai_dir)

# Set it to use GPU1
torch.cuda.set_device(1)
print(torch.cuda.is_available())
print(torch.cuda.current_device())

class AllModels(nn.Module):
    """Fast ai models will all have 2 outputs, so the final will have 2*9. 
       We just use linear layer, loss handles softmax"""
    def __init__(self, num_classes):
        super(AllModels, self).__init__()
        
        models_arch = [resnet34, resnet50, resnet101, vgg16, resnext50, resnext101, inceptionresnet_2, dn121, dn169]
        models_name = ['resnet34', 'resnet50', 'resnet101', 'vgg16', 'resnext50', 'resnext101', 'inceptionresnet_2', 'dn121', 'dn169']
        sz = 224
        PATH = '/media/rene/Data/camelyon_out/tiles_224_100t'
        self.models = {}
        
        for idx, arch in enumerate(models_arch):
            tfms = tfms_from_model(arch, sz, aug_tfms=transforms_top_down, max_zoom=1)
            data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=2)
            learn = ConvLearner.pretrained(arch, data, precompute=False)
            self.models[idx] = learn.model
        self.fc = nn.Linear(9*int(num_classes), int(num_classes))

    def forward(self, x):
        self.int_output = {}
        for key, model in self.models.items():
            self.int_output[key] = model(x)

        x = torch.cat(list(self.int_output.values()), 1)
        print(x.size())
        x = self.fc(x)
        return x

sz=224
PATH = '/media/rene/Data/camelyon_out/tiles_224_1t'
arch = resnet50

tfms = tfms_from_model(arch, sz, aug_tfms=transforms_top_down, max_zoom=1)
data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=64)
# use transforms for resnet 50 for all models
model = AllModels()
learn = ConvLearner.from_model_data(model, data)

learn.crit = nn.CrossEntropyLoss()
learn.metrics = [accuracy]

lrf=learn.lr_find(start_lr=1e-5, end_lr=100)
learn.sched.plot(n_skip=5, n_skip_end=1)

lr = .5
get_ipython().run_line_magic('time', "learn.fit(lr, 3, cycle_len=1, cycle_mult=2, best_save_name='All_9_models_no_unfreeze_1t')")

lr = .05
get_ipython().run_line_magic('time', "learn.fit(lr, 3, cycle_len=1, cycle_mult=2, best_save_name='All_9_models_no_unfreeze_1t')")

lr = .01
get_ipython().run_line_magic('time', "learn.fit(lr, 10, best_save_name='All_9_models_no_unfreeze_1t')")
plt.figure()
learn.sched.plot_loss()
plt.figure()
learn.sched.plot_lr()

lr = .001
get_ipython().run_line_magic('time', "learn.fit(lr, 10, best_save_name='All_9_models_no_unfreeze_1t')")
plt.figure()
learn.sched.plot_loss()
plt.figure()
learn.sched.plot_lr()



sz=299 # so the inception models work
PATH = '/media/rene/Data/camelyon_out/tiles_224_100t'
arch = resnet50

tfms = tfms_from_model(arch, sz, aug_tfms=transforms_top_down, max_zoom=1)
data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=32)
# use transforms for resnet 50 for all models
model = AllModels()
learn = ConvLearner.from_model_data(model, data)

lr = .001
learn.crit = nn.CrossEntropyLoss()
learn.metrics = [accuracy]

get_ipython().run_line_magic('time', "learn.fit(lr, 3, cycle_len=1, cycle_mult=2, best_save_name='All_9_models_no_unfreeze_training')")

sz=299 # so the inception models work
PATH = '/media/rene/Data/camelyon_out/tiles_224_100t'
arch = resnet50

tfms = tfms_from_model(arch, sz, aug_tfms=transforms_top_down, max_zoom=1)
data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=4)
# use transforms for resnet 50 for all models
model = AllModels()
learn = ConvLearner.from_model_data(model, data)

lr = .001
learn.crit = nn.CrossEntropyLoss()
learn.metrics = [accuracy]
learn.unfreeze()
get_ipython().run_line_magic('time', "learn.fit(lr, 3, cycle_len=1, cycle_mult=2, best_save_name='All_9_models_Actually_fully_training')")

# CUDA_LAUNCH_BLOCKING=1
torch.cuda.synchronize()

PATH = "/media/rene/Data/data/cifar10/sample"
num_workers = 4
bs=2
sz=224
# stats = (np.array([ 0.4914 ,  0.48216,  0.44653]), np.array([ 0.24703,  0.24349,  0.26159]))
# tfms = tfms_from_stats(stats, sz, aug_tfms=[RandomFlip()], pad=sz//8)

arch = resnet50
tfms = tfms_from_model(arch, sz, aug_tfms=transforms_top_down, max_zoom=1)

data = ImageClassifierData.from_paths(PATH, val_name='test', tfms=tfms, bs=bs)

model = AllModels(num_classes=10)
learn = ConvLearner.from_model_data(model, data)

lr = 1
wd=1e-4
get_ipython().run_line_magic('time', 'learn.fit(lr, 1, wds=wd, cycle_len=30, use_clr_beta=(20, 20, 0.95, 0.85))')

def train_model(model, model_list, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
        
    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in tqdm(dataloaders[phase]):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
                    
                ######### Get model outputs
                features = []
                for model in model_list:
                    output = model(inputs)
                    features.append(output)
                cat_features = torch.cat(features, 1)
                    
                ###########
                    
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = fc_net(cat_features)

                # for nets that have multiple outputs such as inception
                if isinstance(outputs, tuple):
                    loss = sum((criterion(o,labels) for o in outputs))
                else:
                    loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    _, preds = torch.max(outputs.data, 1)
                    loss.backward()
                    optimizer.step()
                else:
                    _, preds = torch.max(outputs.data, 1)

                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(9*2, 10)
        self.fc2_bn = nn.BatchNorm2d(10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        out1 = F.relu(self.fc2_bn(self.fc1(x)))
        out2 = self.fc2(out1)
        return out2
    
    
fc_net = FCNet()
model = fc_net.cuda()

from torch.optim import lr_scheduler

batch_size = 64
num_workers = 4
PATH = '/media/rene/Data/camelyon_out/tiles_224_100t'


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(PATH, x),
                                          data_transforms[x])
                  for x in ['train', 'valid']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers)
              for x in ['train', 'valid']}

dataset_sizes = {x: len(image_datasets[x])/10 for x in ['train', 'valid']}
class_names = image_datasets['valid'].classes

sz = 224
PATH = '/media/rene/Data/camelyon_out/tiles_224_100t'

# models_arch = [resnet34, resnet50, resnet101, vgg16, resnext50, resnext101, inceptionresnet_2, dn121, dn169]
# for arch in models_arch:
#     tfms = tfms_from_model(arch, sz, aug_tfms=transforms_top_down, max_zoom=1)
#     data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=2)
#     learn = ConvLearner.pretrained(arch, data, precompute=False)
#     model_list.append(learn.model)

# loc = '/media/rene/Data/camelyon_out/tiles_224_100t/models'
models_name = ['resnet34_full_0', 'resnet50_full_0', 'resnet101_full_0', 
               'vgg16_full', 'resnext50_full', 'resnext101_full', 'inceptionresnet_2_full',
               'dn121_full', 'dn169_full_0']

models_arch = [resnet34, resnet50, resnet101, vgg16, resnext50, resnext101, inceptionresnet_2, dn121, dn169]
# models_name = ['resnet34', 'resnet50', 'resnet101', 'vgg16', 'resnext50', 'resnext101', 'inceptionresnet_2', 'inception_4', 'dn121', 'dn169']


model_list = []
for idx, arch in enumerate(models_arch):
    tfms = tfms_from_model(models_arch, sz, aug_tfms=transforms_top_down, max_zoom=1)
    data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=4)
    
    model_loc = os.path.join(PATH, 'models', models_name[idx])
    learn = ConvLearner.pretrained(arch, data, precompute=False)
    learn.load(model_loc)
    model_list.append(learn.model)

num_epochs = 20
save_path = '/media/rene/Data/camelyon/output/tmp/ensemble_model3'
    

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

model = train_model(model, model_list, 
                    criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)
torch.save(model.state_dict(), save_path)

# with open('/media/rene/Data/camelyon_out/tiles_224_100t/tmp/output.txt', 'w') as out:
#     out.write(cap.stdout)



from torch.optim import lr_scheduler

batch_size = 64
num_workers = 4
PATH = '/media/rene/Data/camelyon_out/tiles_224_100t'


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(PATH, x),
                                          data_transforms[x])
                  for x in ['train', 'valid']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers)
              for x in ['train', 'valid']}

dataset_sizes = {}
dataset_sizes['train'] = len(image_datasets['train'])/10
dataset_sizes['valid'] = len(image_datasets['valid'])/10

class_names = image_datasets['valid'].classes

sz = 224
PATH = '/media/rene/Data/camelyon_out/tiles_224_100t'

# models_arch = [resnet34, resnet50, resnet101, vgg16, resnext50, resnext101, inceptionresnet_2, dn121, dn169]
# for arch in models_arch:
#     tfms = tfms_from_model(arch, sz, aug_tfms=transforms_top_down, max_zoom=1)
#     data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=2)
#     learn = ConvLearner.pretrained(arch, data, precompute=False)
#     model_list.append(learn.model)

# loc = '/media/rene/Data/camelyon_out/tiles_224_100t/models'
models_name = ['resnet34_full_0', 'resnet50_full_0', 'resnet101_full_0', 
               'vgg16_full', 'resnext50_full', 'resnext101_full', 'inceptionresnet_2_full',
               'dn121_full', 'dn169_full_0']

models_arch = [resnet34, resnet50, resnet101, vgg16, resnext50, resnext101, inceptionresnet_2, dn121, dn169]
# models_name = ['resnet34', 'resnet50', 'resnet101', 'vgg16', 'resnext50', 'resnext101', 'inceptionresnet_2', 'inception_4', 'dn121', 'dn169']


model_list = []
for idx, arch in enumerate(models_arch):
    tfms = tfms_from_model(models_arch, sz, aug_tfms=transforms_top_down, max_zoom=1)
    data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=4)
    
    model_loc = os.path.join(PATH, 'models', models_name[idx])
    learn = ConvLearner.pretrained(arch, data, precompute=False)
    learn.load(model_loc)
    model_list.append(learn.model)

num_epochs = 2
save_path = '/media/rene/Data/camelyon/output/tmp/ensemble_model2'
    

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

model = train_model(model, model_list, 
                    criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)
torch.save(model.state_dict(), save_path)

# with open('/media/rene/Data/camelyon_out/tiles_224_100t/tmp/output.txt', 'w') as out:
#     out.write(cap.stdout)







image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_loc, x),
                                          data_transforms[x])
                  for x in ['train', 'valid']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.num_workers)
              for x in ['train', 'valid']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes




ensemble_dataloader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.num_workers)
              for x in ['train', 'valid']}




class EnsembleDataloader(Dataset):
    """Ensemble dataset."""
    def __init__(self, PATH, models, transform=None):
        self.PATH = PATH
        self.models = models
        self.transform = transform
        self.all_data = glob.glob(PATH+'/**/*.png', recursive=True)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_loc, x),
                                                  data_transforms[x])
                          for x in ['train', 'valid']}

        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                                     shuffle=True, num_workers=args.num_workers)
                      for x in ['train', 'valid']}

        
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


num_workers = 4
bs = 2
    
ensemble_datasets = {x: EnsembleDataloader(os.path.join(args.data_loc, x))
                  for x in ['train', 'valid']}

ensemble_dataloader = {x: torch.utils.data.DataLoader(ensemble_datasets[x], batch_size=bs,
                                             shuffle=False, num_workers=num_workers)
              for x in ['train', 'valid']}


model_ft = models.resnet50(pretrained=True)

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()

# all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                   num_epochs=args.epochs)
torch.save(model_ft.state_dict(), args.save_path)







