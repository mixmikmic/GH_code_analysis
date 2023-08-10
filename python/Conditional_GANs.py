import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image

def to_cuda(x):
    x = Variable(x)
    if torch.cuda.is_available():
        return x.cuda()
    return x

def to_onehot(x, num_classes=10):
    assert isinstance(x, int) or isinstance(x.data, (torch.cuda.LongTensor, torch.LongTensor))    
    if isinstance(x, int):
        c = to_cuda(torch.zeros(1, num_classes))
        c.data[0][x] = 1
    else:
        c = to_cuda(torch.FloatTensor(x.size(0), num_classes))
        c.zero_()
        c.scatter_(1, x, 1) # dim, index, src value
    return c

class Discriminator(nn.Module):
    """
        Simple Discriminator w/ MLP
    """
    def __init__(self, input_size=784, label_size=10, num_classes=1):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size+label_size, 200),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(200, num_classes),
            nn.Sigmoid(),
        )
    
    def forward(self, x, y):        
        x, y = x.view(x.size(0), -1), y.view(y.size(0), -1)
        v = torch.cat((x, y), 1) # v: [input, label] concatenated vector
        y_ = self.layer1(v)
        y_ = self.layer2(y_)
        y_ = self.layer3(y_)
        return y_

class Generator(nn.Module):
    """
        Simple Generator w/ MLP
    """
    def __init__(self, input_size=100, label_size=10, num_classes=784):
        super(Generator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size+label_size, 200),
            nn.LeakyReLU(0.2),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.2),
            nn.Linear(200, num_classes),
            nn.Tanh()
        )
        
    def forward(self, x, y):
        x, y = x.view(x.size(0), -1), y.view(y.size(0), -1)
        v = torch.cat((x, y), 1) # v: [input, label] concatenated vector
        y_ = self.layer(v)
        y_ = y_.view(x.size(0), 1, 28, 28)
        return y_

D = Discriminator().cuda()
G = Generator().cuda()
D.load_state_dict('D_c.pkl')
G.load_state_dict('G_c.pkl')

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                std=(0.5, 0.5, 0.5))]
)

mnist = datasets.MNIST(root='../data/', train=True, transform=transform, download=True)

batch_size = 64
condition_size = 10

data_loader = DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True, drop_last=True)

criterion = nn.BCELoss()
D_opt = torch.optim.Adam(D.parameters())
G_opt = torch.optim.Adam(G.parameters())

max_epoch = 200 # need more than 500 epochs for training generator
step = 0
n_critic = 5 # for training more k steps about Discriminator
n_noise = 100

D_labels = to_cuda(torch.ones(batch_size)) # Discriminator Label to real
D_fakes = to_cuda(torch.zeros(batch_size)) # Discriminator Label to fake

for epoch in range(max_epoch):
    for idx, (images, labels) in enumerate(data_loader):
        step += 1
        # Training Discriminator
        x = to_cuda(images)
        y = to_cuda(labels)
        y = y.view(batch_size, 1)
        y = to_onehot(y)
        x_outputs = D(x, y)
        D_x_loss = criterion(x_outputs, D_labels)

        z = to_cuda(torch.randn(batch_size, n_noise))
        z_outputs = D(G(z, y), y)
        D_z_loss = criterion(z_outputs, D_fakes)
        D_loss = D_x_loss + D_z_loss
        
        D.zero_grad()
        D_loss.backward()
        D_opt.step()

        if step % n_critic == 0:
            # Training Generator
            z = to_cuda(torch.randn(batch_size, n_noise))
            z_outputs = D(G(z, y), y)
            G_loss = criterion(z_outputs, D_labels)

            D.zero_grad()
            G.zero_grad()
            G_loss.backward()
            G_opt.step()
        
        if step % 1000 == 0:
            print('Epoch: {}/{}, Step: {}, D Loss: {}, G Loss: {}'.format(epoch, max_epoch, step, D_loss.data[0], G_loss.data[0]))        

def save_checkpoint(state, file_name='checkpoint.pth.tar'):
    torch.save(state, file_name)

# Saving params.
# torch.save(D.state_dict(), 'D_c.pkl')
# torch.save(G.state_dict(), 'G_c.pkl')
save_checkpoint({'epoch': epoch + 1, 'state_dict':D.state_dict(), 'optimizer' : D_opt.state_dict()}, 'D_c.pth.tar')
save_checkpoint({'epoch': epoch + 1, 'state_dict':G.state_dict(), 'optimizer' : G_opt.state_dict()}, 'G_c.pth.tar')

import scipy.misc

# generation to image
z = to_cuda(torch.randn(1, n_noise))
y = to_onehot(9)
fakeimg = G(z, y).view(28, 28)
img = fakeimg.cpu().data.numpy()
scipy.misc.toimage(img)



