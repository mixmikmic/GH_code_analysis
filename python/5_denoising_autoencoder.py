import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

cuda = torch.cuda.is_available() # True if cuda is available, False otherwise
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
print('Training on %s' % ('GPU' if cuda else 'CPU'))

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.1307,), (0.3081,))])
mnist = torchvision.datasets.MNIST(root='../data/', train=True, transform=transform, download=True)

batch = 300
data_loader = torch.utils.data.DataLoader(mnist, batch_size=batch, shuffle=True)

autoencoder = nn.Sequential(
                # Encoder
                nn.Linear(28 * 28, 512),
                nn.PReLU(512),
                nn.BatchNorm1d(512),
    
                # Low-dimensional representation
                nn.Linear(512, 128),   
                nn.PReLU(128),
                nn.BatchNorm1d(128),
    
                # Decoder
                nn.Linear(128, 512),
                nn.PReLU(512),
                nn.Linear(512, 28 * 28))

autoencoder = autoencoder.type(FloatTensor)

optimizer = torch.optim.Adam(params=autoencoder.parameters(), lr=0.005)

epochs = 10
data_size = int(mnist.train_labels.size()[0])

for i in range(epochs):
    for j, (images, _) in enumerate(data_loader):
        images = images.view(images.size(0), -1).type(FloatTensor)
        images_noisy = images + 0.2 * torch.randn(images.size()).type(FloatTensor) # adding noise
        images = Variable(images, requires_grad=False)
        images_noisy = Variable(images_noisy)

        autoencoder.zero_grad()
        reconstructions = autoencoder(images_noisy) # forward noisy images
        loss = torch.dist(images, reconstructions) # compare reconstructions to unperturbed images
        loss.backward()
        optimizer.step()
    print('Epoch %i/%i loss %.4f' % (i + 1, epochs, loss.data[0]))

images = images.view(batch, 28, 28).data.cpu().numpy()
reconstructions = reconstructions.view(batch, 28, 28).data.cpu().numpy()

fig = plt.figure(figsize=(6, 6))
for i in range(10):
    a = fig.add_subplot(10, 10, i + 1)
    b = fig.add_subplot(10, 10, i + 11)
    a.axis('off')
    b.axis('off')
    image = images[i - 1]
    reconstruction = reconstructions[i - 1]
    a.imshow(image, cmap='Greys_r')
    b.imshow(reconstruction, cmap='Greys_r');

