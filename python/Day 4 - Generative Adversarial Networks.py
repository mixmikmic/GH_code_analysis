get_ipython().run_line_magic('matplotlib', 'notebook')
import torch
from torch.autograd import Variable
import torchvision
from torchvision import transforms, datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt    

batch_size = 100

train_data = datasets.FashionMNIST(root='fashiondata/',
                                 transform=transforms.ToTensor(),
                                 train=True,
                                 download=True
                                 )

train_samples = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size,
                                           shuffle=True
                                            )

class discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1) #1x28x28-> 64x14x14
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) #64x14x14-> 128x7x7
        self.dense1 = torch.nn.Linear(128*7*7, 1)

        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(128)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))).view(-1, 128*7*7)
        x = F.sigmoid(self.dense1(x))
        return x

class generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(128, 256)
        self.dense2 = torch.nn.Linear(256, 1024)
        self.dense3 = torch.nn.Linear(1024, 128*7*7)
        self.uconv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) #128x7x7 -> 64x14x14
        self.uconv2 = torch.nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1) #64x14x14 -> 1x28x28

        self.bn1 = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(1024)
        self.bn3 = torch.nn.BatchNorm1d(128*7*7)
        self.bn4 = torch.nn.BatchNorm2d(64)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.dense1(x)))
        x = F.relu(self.bn2(self.dense2(x)))
        x = F.relu(self.bn3(self.dense3(x))).view(-1, 128, 7, 7)
        x = F.relu(self.bn4(self.uconv1(x)))
        x = F.sigmoid(self.uconv2(x))
        return x

#instantiate model
d = discriminator()
g = generator()

#training hyperparameters
no_epochs = 100
dlr = 0.0003
glr = 0.0003

d_optimizer = torch.optim.Adam(d.parameters(), lr=dlr)
g_optimizer = torch.optim.Adam(g.parameters(), lr=glr)

dcosts = []
gcosts = []
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Epoch')
ax.set_ylabel('Cost')
ax.set_xlim(0, no_epochs)
plt.show()

def train(no_epochs, glr, dlr):
    for epoch in range(no_epochs):
        epochdcost = 0
        epochgcost = 0

        #iteratre over mini-batches
        for k, (real_images, _ ) in enumerate(train_samples):
            real_images = Variable(real_images) #real images from training set

            z = Variable(torch.randn(batch_size, 128)) #generate random latent variable to generate images
            generated_images = g.forward(z) #generate images

            gen_pred = d.forward(generated_images) #prediction of generator on generated batch
            real_pred = d.forward(real_images) #prediction of generator on real batch

            dcost = -torch.sum(torch.log(real_pred)) - torch.sum(torch.log(1-gen_pred)) #cost of discriminator
            gcost = -torch.sum(torch.log(gen_pred))/batch_size #cost of generator
            
            #train discriminator
            d_optimizer.zero_grad()
            dcost.backward(retain_graph=True) #retain the computation graph so we can train generator after
            d_optimizer.step()
            
            #train generator
            g_optimizer.zero_grad()
            gcost.backward()
            g_optimizer.step()

            epochdcost += dcost.data[0]
            epochgcost += gcost.data[0]
            
            #give us an example of a generated image after every 10000 images produced
            if k*batch_size%10000 ==0:
                g.eval() #put in evaluation mode
                noise_input = Variable(torch.randn(1, 128))
                generated_image = g.forward(noise_input)

                plt.figure(figsize=(1, 1))
                plt.imshow(generated_image.data[0][0], cmap='gray_r')
                plt.show()
                g.train() #put back into training mode


            epochdcost /= 60000/batch_size
            epochgcost /= 60000/batch_size

            print('Epoch: ', epoch)
            print('Disciminator cost: ', epochdcost)
            print('Generator cost: ', epochgcost)
        
        #plot costs
        
        dcosts.append(epochdcost)
        gcosts.append(epochgcost)

        ax.plot(dcosts, 'b')
        ax.plot(gcosts, 'r')

        fig.canvas.draw()

train(no_epochs, glr, dlr)

