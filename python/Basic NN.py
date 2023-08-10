# Imports
# Torch package
import torch
# Variable Class
from torch.autograd import Variable
# Neural Net Sub-package from Torch
import torch.nn as nn
# Functions from the Neural Net Sub-Package such as RELU
import torch.nn.functional as F

# input_size will dictate size of first hidden layer
# Here, have 3 samples with 8 coordinates each
inputs = Variable(torch.randn(3,8), requires_grad=True)
input_size = inputs.size()

# Implement the Neural Net class from torch.nn.Module Class
# Has all the useful stuff needed for a Neural Net
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        # Basic net architecture i.e. the layers needed in all
        # How they are chained and non-linearity addition defined in forward()
        # A layer is defined by number of inputs and outputs
        # Input layer already specified by size of input fed when instantiating the net
        self.hidden = nn.Linear(input_size[1], 5)
        self.output = nn.Linear(5, 2)

    def forward(self, inputs):
        # How lyers link to each other is defined here
        # Non-linearity added in between this chaining definition
        
        # Input 3 samples of size 8 into the hidden layer
        x = self.hidden(inputs)
        x = F.tanh(x)
        x = self.output(x)
        x = F.log_softmax(x)
        return x

    # Can also define any other helpers to be used by this NN here
    
# Instantiate Net architecture first
net = Net()
print 'This net looks like: ', net
# Calling the net on a set of inputs does calls forward() i.e. forward props on all of them,
# Useful if say you have a trained net and just need to classify test data
result = net(inputs)
print 'Classifying the inputs as if it is a trained model gives: ', result

# In one epoch:
#  The net forward propagates on the input
#  Calculates loss using the loss function on outputs obtained and desired_output
#  Optimizes the loss function using the optimizer
#    That is, finds gradient of loss wrt. each weight (Recall the Variable Class that wraps around a Tensor)
#    (SGD is an example seen in theory)
#  Updates the weights based on those gradients

def feed_forward_one_time(net, inputs, desired_outputs):
    # Forward prop
    output = net(inputs)
    
    # Calculate loss
    loss_function = nn.MSELoss()
    loss = loss_function(output, desired_outputs)
    print('Loss is now valued at: ', loss)
    return loss

net = Net()
loss = feed_forward_one_time(net, inputs, Variable(torch.rand(3, 2)))

# Note: Both output and desired output ned to be Variables with same type of tensor inside
# Cast a tensor by doing that_tensor_name.float() or .double() or .long() etc etc

# Clear all gradient buffers for params to get fresh gradients
net.zero_grad() 
# Backprop and get gradient for each and every param
loss.backward()

# Loss is a Variable that has its grad_fn spanning all the way back to inputs 
    # This allows Backpropagation wrt. every parameter
    # Each parameter is in the net.parameters() generator:
layer_count = 0
for x in net.parameters():
    print 'layer #', layer_count, 'Parameters'
    print x.data
    print 'layer #', layer_count, 'Gradients'
    print x.grad.data
    layer_count += 1

# Now can update all those seen parameters using the gradients obtained
def update_parameters(net, learning_rate = 0.01):
    for f in net.parameters():
        f.data.sub_(f.grad.data * learning_rate)

print 'weights from input layer to hidden BEFORE update:'
print net.parameters().next().data
update_parameters(net)
print 'weights from input layer to hidden AFTER update:'
print net.parameters().next().data

import torch.optim as optim
net = Net()
# Choose loss function, optimizer
loss_function = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# Loop through:
loss_value = 1000000
update_count = 0
while loss_value > 1:
    # Regular feed-forward and loss calculation
    output = net(inputs)
    loss = loss_function(output, Variable(torch.rand(3, 2)))
    loss_value = loss.data[0]
    # Zero_grad on optimizer directly which now wraps around the net params
    optimizer.zero_grad()
    # Backpropagate
    loss.backward()
    # Adjust weights based on how this optimizer does it in theory
    optimizer.step()
    update_count += 1
    print 'weight from input layer neurone 1 to hidden layer neurone 1 after update #', update_count, ': '
    print net.parameters().next().data[0,0]
    
print 'Loss reached in the end: ', loss_value

