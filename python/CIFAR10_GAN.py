get_ipython().magic('matplotlib inline')
#Import a ton of stuff
import pickle
import numpy as np

import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from layers import Conv2d, ConvTranspose2d, Linear
from InfoGAN import InfoGAN

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

directory = './CIFAR_Data/cifar-10-batches-py/'

data_train = []
targets_train = []
for i in range(5):
    raw = unpickle(directory + 'data_batch_' + str(i + 1))
    data_train += [raw[b'data']]
    targets_train += raw[b'labels']

data_train = np.concatenate(data_train)
targets_train = np.array(targets_train)

data_test = unpickle(directory + 'test_batch')
targets_test = np.array(data_test[b'labels'])
data_test = np.array(data_test[b'data'])

print(data_train.shape, targets_train.shape)

# Now reformat it into the format we want
# NOTE: PyTorch is weird so if we wanted to use cross entropy we need to keep them as logits, but we won't so...
x_train = np.reshape(data_train, (-1, 3, 32, 32))
y_train = np.zeros((targets_train.shape[0], 10), dtype = np.uint8)
y_train[np.arange(targets_train.shape[0]), targets_train] = 1

x_test = np.reshape(data_test, (-1, 3, 32, 32))
y_test = np.zeros((targets_test.shape[0], 10), dtype = np.uint8)
y_test[np.arange(targets_test.shape[0]), targets_test] = 1

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# Looks like we can convert our x to unint8
# We'll convert back to float and rescale to between 0 and 1 on the GPU batchwise to save CPU RAM
print(np.unique(x_train))

# Show an example from the test set
print(y_test[0])
plt.figure(0)
plt.imshow(np.transpose(x_test[0], (1, 2, 0)), cmap = 'gray')

supervision = 1000 # Number of samples to supervise with

# Prep the data by turning them into tensors and putting them into a PyTorch dataloader
shuffle_train = np.random.permutation(y_train.shape[0])
x_train_th = torch.from_numpy(x_train[shuffle_train])
y_train_th = torch.from_numpy(y_train[shuffle_train]).float()

x_test_th = torch.from_numpy(x_test)
y_test_th = torch.from_numpy(y_test)

# OK, we're going to be hacking this out. We'll multiply by the sum of the labels
# So to make this semisupervised, we set the labels we don't want to 0
y_train_th[int(supervision):] = 0

train_tensors = TensorDataset(x_train_th, y_train_th)
test_tensors = TensorDataset(x_test_th, y_test_th)
train_loader = DataLoader(train_tensors, batch_size = 128, shuffle = True, num_workers = 6, pin_memory = True)
test_loader = DataLoader(test_tensors, batch_size = 128, shuffle = True, num_workers = 6, pin_memory = True)

# Now let's start building the GAN
# But first, we're going to redefine Conv2D and Linear with our own initialisations
# We're going to use Glorot (aka Xavier) uniform init for all weights
# And we will use zero init for all biases

c1_len = 10 # Multinomial
c2_len = 0 # Gaussian
c3_len = 0 # Bernoulli
z_len = 64 # Noise vector length
embedding_len = 128

class Conv2d(nn.Conv2d):
    def reset_parameters(self):
        stdv = np.sqrt(6 / ((self.in_channels  + self.out_channels) * np.prod(self.kernel_size)))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

class ConvTranspose2d(nn.ConvTranspose2d):
    def reset_parameters(self):
        stdv = np.sqrt(6 / ((self.in_channels  + self.out_channels) * np.prod(self.kernel_size)))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

class Linear(nn.Linear):
    def reset_parameters(self):
        stdv = np.sqrt(6 / (self.in_features + self.out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = Linear(z_len + c1_len + c2_len + c3_len, 1024)
        self.fc2 = Linear(1024, 4 * 4 * 256)

        self.convt1 = ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 1)
        self.convt2 = ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 1)
        self.convt3 = ConvTranspose2d(64, 3, kernel_size = 4, stride = 2, padding = 1)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(4 * 4 * 256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x))).view(-1, 256, 4, 4)

        x = F.relu(self.bn3(self.convt1(x)))
        x = F.relu(self.bn4(self.convt2(x)))
        x = self.convt3(x)

        return F.sigmoid(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = Conv2d(3, 64, kernel_size = 4, stride = 2, padding = 1) # 32 x 32 -> 16 x 16
        self.conv2 = Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1) # 16 x 16 -> 8 x 8
        self.conv3 = Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1) # 8 x 8 -> 4 x 4

        self.fc1 = Linear(256 * 4 ** 2, 1024)
        self.fc2 = Linear(1024, 1)
        self.fc1_q = Linear(1024, embedding_len)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn_q1 = nn.BatchNorm1d(embedding_len)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.bn1(self.conv2(x)))
        x = F.leaky_relu(self.bn2(self.conv3(x))).view(-1, self.fc1.in_features)

        x = F.leaky_relu(self.bn3(self.fc1(x)))
        return self.fc2(x), F.leaky_relu(self.bn_q1(self.fc1_q(x)))

# OK, now we create the actual models
gen = Generator().cuda()
dis = Discriminator().cuda()

# Link it all together into the InfoGAN. Also add the output layers for the latent codes
gan = InfoGAN(gen, dis, embedding_len, z_len, c1_len, c2_len, c3_len)

# Alright, everything's setup, let's run the GAN and train it
gan.train_all(train_loader)

gan.save('./CIFAR/')

gan.load('./CIFAR/')

plt.figure(0, figsize = (32, 32))

z_dict = gan.get_z(10 * c1_len, sequential = True)
out_gen = gan.gen(torch.cat([z_dict[k] for k in z_dict.keys()], dim = 1))

for i in range(10):
    for j in range(10):
        idx = i * 10 + j + 1
        plt.subplot(10, 10, idx)
        plt.imshow(np.transpose(out_gen[idx - 1].cpu().data.numpy(), (1, 2, 0)), cmap = 'gray')

out_test = gan.run_dis(Variable(x_test_th).cuda().float() / 255)[1]
out_test = np.argmax(out_test.data.cpu().numpy(), axis = 1)

print(np.mean(out_test == np.argmax(y_test, axis = 1)))



