# import functionality from these libraries
get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np      # for efficient numerical computation
import torch            # for building computational graphs
from torch.autograd import Variable     # for automatically computing gradients of our cost with respect to what we want to optimise
import matplotlib.pyplot as plt     # for plotting absolutely anything
from mpl_toolkits.mplot3d import Axes3D # for plotting 3D graphs
import pandas as pd #allows us to easily import any data

df = pd.read_csv('airfoil_self_noise.dat', sep='\t')#import our dataset into a pandas dataframe
df = df.sample(frac=1) #shuffle our dataset
print(df.head())

#convert our data into torch tensors
X = torch.Tensor(np.array(df[df.columns[0:-1]])) #pick our features from our dataset
Y = torch.Tensor(np.array(df[df.columns[-1]])) #select our label

X = (X-X.mean(0)) / X.std(0) #normalize our features along the 0th axis

m = 1100 #size of training set

#split our data into training and test set
#training set
x_train = Variable(X[0:m])
y_train = Variable(Y[0:m])

#test set
x_test = Variable(X[m:])
y_test = Variable(Y[m:])

#define model class - inherit useful functions and attributes from torch.nn.Module
class linearmodel(torch.nn.Module):
    def __init__(self):
        super().__init__() #call parent class initializer
        self.linear = torch.nn.Linear(5, 1) #define linear combination function with 11 inputs and 1 output

    def forward(self, x):
        x = self.linear(x) #linearly combine our inputs to give 1 outputs
        return x

no_epochs = 100
lr = 10

#create our model from defined class
mymodel = linearmodel()
criterion = torch.nn.MSELoss() #cross entropy cost function as it is a classification problem
optimizer = torch.optim.Adam(mymodel.parameters(), lr = lr) #define our optimizer

#for plotting costs
costs=[]
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Epoch')
ax.set_ylabel('Cost')
ax.set_xlim(0, no_epochs-1)
plt.show()

#training loop - same as last time
def train(no_epochs):
    for epoch in range(no_epochs):
        h = mymodel.forward(x_train) #forward propagate - calulate our hypothesis
        #calculate, plot and print cost
        cost = criterion(h, y_train)
        costs.append(cost.data[0])
        ax.plot(costs, 'b')
        fig.canvas.draw()
        print('Epoch ', epoch, ' Cost: ', cost.data[0])

        #calculate gradients + update weights using gradient descent step with our optimizer
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

train(no_epochs)

def test():
    h = mymodel.forward(x_test)
    cost = criterion(h, y_test)
    
    return cost.data[0]

test_cost = test()
print('Cost on test set: ', test_cost)

