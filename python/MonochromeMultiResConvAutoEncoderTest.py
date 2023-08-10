import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from torchvision import datasets
from torchvision.utils import save_image

import skimage 
import math
# import io
# import requests
# from PIL import Image

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import sys
import os

import cae

from helpers import *
from helper_modules import *
from multi_res_cae import *

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '')

get_ipython().run_line_magic('aimport', 'helpers, helper_modules, multi_res_cae')

get_ipython().run_line_magic('aimport', '')

# frcae = MultiFullCAE((640,480))

# mrcae = MultiResCAE([640,480])

# Hyper Parameters
# num_epochs = 5
# batch_size = 100
# learning_rate = 0.001

num_epochs = 20
batch_size = 128
learning_rate = 0.0001

get_ipython().run_cell_magic('time', '', '\nmodel = MultiFullCAE(in_img_shape=(32,32), full_image_resize=(24,24)).cuda()\n# model = MultiResCAE(in_img_shape=[32,32], channels=3, conv_layer_feat=[16, 32, 64],\n#                  res_px=[[24, 24], [16, 16], [12, 12]], crop_sizes=[[32, 32], [24,24], [12, 12]],\n#                  # conv_sizes = [(3,5,7), (3,5,7,11), (3,5,7,11)]  # this is too much I think\n#                  # conv_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5, 7]]  # test b\n#                  conv_sizes=[[5, 7, 11], [3, 5, 7, 9], [1, 3, 5]]  # test c\n#         ).cuda()')

# model.parameters

get_ipython().run_cell_magic('time', '', 'criterion = nn.MSELoss()\n#criterion = nn.CrossEntropyLoss()\noptimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)')

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 32, 32)
    return x

get_ipython().run_cell_magic('time', '', '\ntransformation = monochrome_preprocess(32,32)\n#transformation = fullimage_preprocess(32,32)\n#train_loader, test_loader = get_loaders(batch_size, transformation, dataset=datasets.CocoDetection)\ntrain_loader, test_loader = get_loaders(batch_size, transformation)')

get_ipython().run_cell_magic('time', '', '\nfor epoch in range(num_epochs):\n    for i, (img, labels) in enumerate(train_loader):\n        img = Variable(img).cuda()\n        # ===================forward=====================\n#         print("encoding batch of  images")\n        output = model(img)\n#         print("computing loss")\n        loss = criterion(output, img)\n        # ===================backward====================\n#         print("Backward ")\n        optimizer.zero_grad()\n        loss.backward()\n        optimizer.step()\n    # ===================log========================\n    print(\'epoch [{}/{}], loss:{:.4f}\'.format(epoch+1, num_epochs, loss.data[0]))\n    if epoch % 4 == 0:\n        pic = to_img(output.cpu().data)\n        in_pic = to_img(img.cpu().data)\n        save_image(pic, \'./fmrcae_results/c_MergingLayer_in-32x32_3-5-7-11_out_image_{}.png\'.format(epoch))\n        save_image(in_pic, \'./fmrcae_results/c_MergingLayer_in-3-5-7-11_in_image_{}.png\'.format(epoch))\n#     if loss.data[0] < 0.21: #arbitrary number because I saw that it works well enough\n#         break')

#torch.save("fmrcae_in-64x64_32x32_3-5-7-11.pth", model)
#torch.save("mrcae_in-32x32_.pth", model)

