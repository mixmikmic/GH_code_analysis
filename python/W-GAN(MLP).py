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

def n_critic(step, nc=5):
    if step < 25 or step % 500 == 0:
        return 100
    return nc

class Discriminator(nn.Module):
    """
        Convolutional Discriminator for MNIST
    """
    def __init__(self, input_size=784, num_classes=1):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 200),
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
        )
    
    def forward(self, x):
        y_ = x.view(x.size(0), -1)
        y_ = self.layer1(y_)
        y_ = self.layer2(y_)
        y_ = self.layer3(y_)
        return y_

class Generator(nn.Module):
    """
        Convolutional Generator for MNIST
    """
    def __init__(self, input_size=100, num_classes=784):
        super(Generator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, num_classes),
            nn.Tanh()
        )
        
    def forward(self, x):
        y_ = self.layer(x)
        y_ = y_.view(x.size(0), 1, 28, 28)
        return y_

D = Discriminator().cuda()
G = Generator().cuda()

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                std=(0.5, 0.5, 0.5))]
)

mnist = datasets.MNIST(root='../data/', train=True, transform=transform, download=True)

batch_size = 64

data_loader = DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True, drop_last=True)

D_opt = torch.optim.RMSprop(D.parameters(), lr=0.00005)
G_opt = torch.optim.RMSprop(G.parameters(), lr=0.00005)

max_epoch = 100 # need more than 100 epochs for training generator
step = 0
g_step = 0
n_noise = 100

D_labels = to_cuda(torch.ones(batch_size)) # Discriminator Label to real
D_fakes = to_cuda(torch.zeros(batch_size)) # Discriminator Label to fake

for epoch in range(max_epoch):
    for idx, (images, labels) in enumerate(data_loader):
        step += 1
                   
        # Training Discriminator
        x = to_cuda(images)
        x_outputs = D(x)

        z = to_cuda(torch.randn(batch_size, n_noise))
        z_outputs = D(G(z))
        D_x_loss = torch.mean(x_outputs)
        D_z_loss = torch.mean(z_outputs)
        D_loss = (D_x_loss - D_z_loss) * -1
        
        D.zero_grad()
        D_loss.backward()
        D_opt.step()
        # Parameter Clipping
        for p in D.parameters():
            p.data.clamp_(-0.01, 0.01)
                    
        if step % n_critic(g_step) == 0:
            g_step += 1
            # Training Generator
            z = to_cuda(torch.randn(batch_size, n_noise))
            z_outputs = D(G(z))
            G_loss = -torch.mean(z_outputs, 0)

            D.zero_grad()
            G.zero_grad()
            G_loss.backward()
            G_opt.step()
        
        if step % 1000 == 0:
            print('Epoch: {}/{}, Step: {}, D Loss: {}, G Loss: {}'.format(epoch, max_epoch, step, -D_loss.data[0], -G_loss.data[0]))

import scipy.misc

# generation to image
z = to_cuda(torch.randn(1, n_noise))
fakeimg = G(z).view(28, 28)
img = fakeimg.cpu().data.numpy()
scipy.misc.toimage(img)

def save_checkpoint(state, file_name='checkpoint.pth.tar'):
    torch.save(state, file_name)

# Saving params.
# torch.save(D.state_dict(), 'D_c.pkl')
# torch.save(G.state_dict(), 'G_c.pkl')
save_checkpoint({'epoch': epoch + 1, 'state_dict':D.state_dict(), 'optimizer' : D_opt.state_dict()}, 'D_was.pth.tar')
save_checkpoint({'epoch': epoch + 1, 'state_dict':G.state_dict(), 'optimizer' : G_opt.state_dict()}, 'G_was.pth.tar')



