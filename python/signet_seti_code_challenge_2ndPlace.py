# Uncomment below and run once to install necessary packages

# !pip install --upgrade http://download.pytorch.org/whl/cu75/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl 
# !pip install torchvision
# !pip install tabulate
# !pip install --upgrade scikit-learn
# !pip install --upgrade numpy
# !pip install --upgrade ibmseti==2.0.0.dev5
# !pip install --upgrade pandas

# Uncomment and run once to install SETI preview test set

# !wget https://dal.objectstorage.open.softlayer.com/v1/AUTH_cdbef52bdf7a449c96936e1071f0a46b/simsignals_v3_zipped/primary_testset_preview_v3.zip
# !unzip -q primary_testset_preview_v3.zip

# Uncomment and run once to download Signet final model parameters

# import requests
# import shutil

# model_url = 'https://dal.objectstorage.open.softlayer.com/v1/AUTH_cdbef52bdf7a449c96936e1071f0a46b/code_challenge_models/signet/final_densenet_model.pth'
# r = requests.get(model_url, stream=True)
# filename = 'signet_final_densenet_model.pth'
# with open(filename, 'wb') as fout:
#     shutil.copyfileobj(r.raw, fout)
# print('saved {}'.format(filename))

get_ipython().system('ls -alrth')

import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict

class DenseNet(nn.Module):
    def __init__(self, pretrained):
        super(DenseNet, self).__init__()
        self.densenet = list(torchvision.models.densenet201(
            pretrained=pretrained
        ).features.named_children())
        self.densenet[0] = ('conv0', nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False))
        self.densenet = nn.Sequential(OrderedDict(self.densenet))
        self.linear = nn.Linear(3840, 7)

    def forward(self, minibatch):
        dense = self.densenet(minibatch)
        out = F.relu(dense, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7).view(dense.size(0), -1)
        return self.linear(out)

import sys

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import ibmseti  #need version 2.0.0 or greater (lastest version: pip install ibmseti==2.0.0.dev5)
import time

def get_spectrogram(filename):
    raw_file = open(filename, 'rb')
    aca = ibmseti.compamp.SimCompamp(raw_file.read())
    tensor = torch.from_numpy(aca.get_spectrogram()).float().view(1, 1, 384, 512)
    return Variable(tensor, volatile=True)

def get_densenet(model_path):
    dense = DenseNet(False).cpu()
    dense.eval()
    state = torch.load(model_path, map_location=lambda storage, loc: storage)
    dense.load_state_dict(state['model'])
    return dense
    

get_ipython().magic("time model = get_densenet('signet_final_densenet_model.pth')")

simfile = 'primary_testset_preview_v3/00b3b8fdb14ce41f341dbe251f476093.dat'

get_ipython().magic('time spec = get_spectrogram(simfile)')

get_ipython().magic('time results = F.softmax(model(spec)).data.view(7)')

import numpy as np
probs = np.array(results.tolist())

print('final class probabilities')
print(probs)

class_list = ['brightpixel', 'narrowband', 'narrowbanddrd', 'noise', 'squarepulsednarrowband', 'squiggle', 'squigglesquarepulsednarrowband']
print('signal classification')
predicted_signal_class = class_list[probs.argmax()]
print(predicted_signal_class)

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

aca = ibmseti.compamp.SimCompamp(open(simfile,'rb').read())
spectrogram = aca.get_spectrogram()
fig, ax = plt.subplots(figsize=(20, 10))
ax.imshow(np.log(spectrogram),  aspect = 0.5*float(spectrogram.shape[1]) / spectrogram.shape[0], cmap='gray')

import pandas as pd
preview_test_set_pd = pd.read_csv('https://github.com/setiQuest/ML4SETI/raw/master/results/private_list_primary_v3_testset_preview_uuid_class_29june_2017.csv', index_col=None)

expected_signal_class = preview_test_set_pd[preview_test_set_pd.UUID == simfile.split('/')[-1].rstrip('.dat')].SIGNAL_CLASSIFICATION.values[0]

assert predicted_signal_class == expected_signal_class
print(expected_signal_class)



