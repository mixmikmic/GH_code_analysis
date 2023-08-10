import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import cv2
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

get_ipython().system('mkdir data/karen')

get_ipython().system('curl https://scontent-nrt1-1.cdninstagram.com/vp/9feb62bac4fd1232dc9fa06d8ef60a2b/5B7D4456/t51.2885-15/e35/18380827_208083853041745_3411566881981595648_n.jpg -o data/karen/karen_hr.jpg')

get_ipython().system('ls data/karen')



hr_img = cv2.imread('./data/karen/karen_hr.jpg')
hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
# hr_img = hr_img[80:360, 380:660]
plt.figure(figsize = (15, 15))
plt.imshow(hr_img)
plt.show()

lr_img = cv2.resize(hr_img, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_CUBIC)
plt.figure(figsize = (15, 15))
plt.imshow(lr_img)
plt.show()



def createDownscaledImage (hr_img, scalingFactor = 0.1):
    return cv2.resize(hr_img, None, fx = scalingFactor, fy = scalingFactor, interpolation = cv2.INTER_CUBIC)

def showImage (img):
    show_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize = (10, 10))
    plt.imshow(show_img)
    plt.show()

cv2.imwrite("./data/karen/karen_hr.jpg", hr_img)
for index in range(10):
    scaling = (index + 1) / 10.0
    im = createDownscaledImage(hr_img, scaling)
    cv2.imwrite("./data/karen/karen-{}.jpg".format(index), im)

hr_img.shape

lr_img = createDownscaledImage(hr_img, 0.5)

lr_img.shape

def imshow (output):
    plt_img = np.clip(output.data.numpy()[0].transpose((1, 2, 0)), 0, 1)
    plt.imshow(plt_img)
    plt.show()

class DeepImagePriorSR(nn.Module):
    def __init__(self, input_noise, width, height, down):
        super(DeepImagePriorSR, self).__init__()

        self.input_noise = input_noise
        self.width = width
        self.height = height

        self.layer = nn.Sequential(
            nn.Conv2d(input_noise, 64, 4, padding=2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 3, 4, padding=1)
        )

        self.d_layer = nn.Sequential(
            nn.MaxPool2d(down)
        )

    def forward(self, x):
        out = self.layer(x)
        out = self.d_layer(out)
        return out

    def hr(self, x):
        out = self.layer(x)
        return out

    def make_z(self):
        return torch.randn([1, self.input_noise, self.width, self.height])

net = DeepImagePriorSR(32, 280, 280, 2)

print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())

input = net.make_z()

output = net(Variable(input))

output.shape

imshow(output)

output.data.numpy()[0].transpose((1, 2, 0)).shape

output = net.hr(Variable(input))
output.shape

train_data = np.array([lr_img]).transpose((0, 3, 1, 2))

train_data.shape

plt.imshow((train_data / 255)[0].transpose((1, 2, 0)))

plt.imshow(lr_img)

def train(DIP, optimizer, criterion, lr_train_img, epoch=10):
    DIP.train()
    
    lr_train = Variable(
        torch.from_numpy(
            (np.array([lr_train_img]) / 255).transpose((0, 3, 1, 2))
        ).float()
    )
    
    DIP_running_loss = 0
    
    for i in range(epoch):
        z = Variable(DIP.make_z())
        
        optimizer.zero_grad()
        
        downed = DIP(z)
        loss = criterion(downed, lr_train)
        
        loss.backward()
        optimizer.step()
        
        DIP_running_loss = loss.data[0]
        
    print("loss: {}".format(DIP_running_loss))
    
    temp_z = Variable(DIP.make_z())
    
    hr = DIP.hr(temp_z)
#     plt.imshow(lr_train_img)
#     plt.show()
    imshow(hr)

DIP_optim = optim.Adam(net.parameters(), 0.003, betas=(0.5, 0.999))
criter = nn.MSELoss()

train(net, DIP_optim, criter, lr_img, 10)

for i in range(5):
    print("epoch {} / 5 - 10 training".format(i + 1))
    train(net, DIP_optim, criter, lr_img, 10)



