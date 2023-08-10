import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Iris.csv')
df[['Species']] = df['Species'].map({'Iris-setosa':0, 'Iris-virginica':1, 'Iris-versicolor':2}) #map text labels to numberical vaules
df = df.sample(frac=1) #shuffle our dataset

print(df.head())

X = torch.Tensor(np.array(df[df.columns[1:-1]])) #pick our features from our dataset
Y = torch.LongTensor(np.array(df[['Species']]).squeeze()) #select our label - squeeze() removes redundant dimensions

m = 100 #size of training set

#training set
x_train = Variable(X[0:m])
y_train = Variable(Y[0:m])

#test set
x_test = Variable(X[m:])
y_test = Variable(Y[m:])

class logisticmodel(torch.nn.Module):
    def __init__(self):
        super().__init__() #call parent class initializer
        self.linear = torch.nn.Linear(4, 3) #define linear combination function with 4 inputs and 3 outputs

    def forward(self, x):
        x = self.linear(x) #linearly combine our inputs to give 3 outputs
        x = torch.nn.functional.softmax(x, dim=1) #activate our output neurons to give probabilities of belonging to each of the three class
        return x

no_epochs = 100
lr = 0.1

mymodel = logisticmodel() #create our model from defined class
criterion = torch.nn.CrossEntropyLoss() #cross entropy cost function as it is a classification problem
optimizer = torch.optim.Adam(mymodel.parameters(), lr = lr) #define our optimizer

costs=[] #store the cost each epoch
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Epoch')
ax.set_ylabel('Cost')
ax.set_xlim(0, no_epochs)
plt.show()

#define train function
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

train(no_epochs) #train the model

test_h = mymodel.forward(x_test) #predict probabilities for test set
_, test_h = test_h.data.max(1) #returns the output which had the highest probability
test_y = y_test.data
correct = torch.eq(test_h, test_y) #perform element-wise equality operation
accuracy = torch.sum(correct)/correct.shape[0] #calculate accuracy
print('Test accuracy: ', accuracy)

torch.save(mynet.state_dict(), 'trained_logistic_model')

