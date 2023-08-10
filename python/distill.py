# %%writefile teacher.py
# %load teacher.py
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable

model = None
optimizer = None
epochs = 20
batch_size = 128
lr = 1e-4

class Teacher(nn.Module):
    def __init__(self):
        super(Teacher, self).__init__()
        self.n_inputs = 28 * 28
        self.n_layer_1 = 1200
        self.n_layer_2 = 1200
        self.n_classes = 10
        self.drop = nn.Dropout(0.5)
        self.affine1 = nn.Linear(self.n_inputs, self.n_layer_1)
        self.affine2 = nn.Linear(self.n_layer_1, self.n_layer_2)
        self.affine3 = nn.Linear(self.n_layer_2, self.n_classes)
    
    def forward(self, x):
        x = x.view(-1, self.n_inputs)
        out1 = self.drop(F.relu(self.affine1(x)))
        out2 = self.drop(F.relu(self.affine2(out1)))
        out3 = self.affine3(out2)
        return out3
    
def train():
    model.train()
    for epoch in xrange(epochs):
        avg_loss = 0
        n_batches = len(train_loader)
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.cuda(), label.cuda()
            data, label = Variable(data), Variable(label)
            optimizer.zero_grad()
            output = F.log_softmax(model(data))
            loss = F.nll_loss(output, label)
            loss.backward()
            optimizer.step()
            avg_loss += loss.data[0]
        avg_loss /= n_batches
        print(avg_loss)

def test():
    model.eval()
    correct = 0
    for data, label in test_loader:
        data, label = data.cuda(), label.cuda()
        data, label = Variable(data, volatile=True), Variable(label)
        output = F.log_softmax(model(data))
        pred = output.data.max(1)[1]
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()
    
    print(100. * correct / len(test_loader.dataset))

    
if __name__ == "__main__":
    model = Teacher()
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)
    train()
    with open('teacher.params', 'wb') as f:
        torch.save(model, f)
    test()
    

# %%writefile student.py
# %load student.py
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable

class Student(nn.Module):
    def __init__(self):
        super(Student, self).__init__()
        self.n_inputs = 28 * 28
        self.n_layer_1 = 800
        self.n_layer_2 = 800
        self.n_classes = 10
        self.drop = nn.Dropout(0.5)
        self.affine1 = nn.Linear(self.n_inputs, self.n_layer_1)
        self.affine2 = nn.Linear(self.n_layer_1, self.n_layer_2)
        self.affine3 = nn.Linear(self.n_layer_2, self.n_classes)
    
    def forward(self, x):
        x = x.view(-1, self.n_inputs)
        out1 = self.drop(F.relu(self.affine1(x)))
        out2 = self.drop(F.relu(self.affine2(out1)))
        out3 = self.affine3(out2)
        return out3

get_ipython().run_cell_magic('writefile', 'teacher-student.py', "# %load teacher-student.py\nfrom __future__ import print_function\nfrom __future__ import division\n\nimport torch\nimport torch.nn as nn\nimport torch.optim as optim\nimport torch.nn.functional as F\nimport torch.utils.data\nimport numpy as np\nfrom torchvision import datasets, transforms\nfrom torch.autograd import Variable\nfrom student import *\nfrom teacher import *\n\nteacher = None\nstudent = None\nepochs = 5\nbatch_size = 64\nlr = 1e-3\n\noptimizer = None\n\ndef train():\n    T = 18\n    student.train()\n    for p in teacher.parameters():\n        p.requires_grad = False\n    for epoch in xrange(epochs):\n        avg_loss = 0\n        n_batches = len(transfer_loader)\n        for data, _ in transfer_loader:\n            data = data.cuda()\n            data = Variable(data)\n            optimizer.zero_grad()\n            output_teacher = F.softmax(teacher(data) / T)\n            output_student = F.softmax(student(data) / T)\n            loss = F.binary_cross_entropy(output_student, output_teacher)\n            loss.backward()\n            optimizer.step()\n            avg_loss += loss.data[0]\n            \n        avg_loss /= n_batches\n        print(avg_loss)\n        \n    \ndef test():\n    T = 1\n    student.eval()\n    correct = 0\n    for data, label in test_loader:\n        data, label = data.cuda(), label.cuda()\n        data, label = Variable(data, volatile=True), Variable(label)\n        output = F.log_softmax(student(data) / T)\n        pred = output.data.max(1)[1]\n        correct += pred.eq(label.data.view_as(pred)).cpu().sum()\n    \n    print(100. * correct / len(test_loader.dataset))\n\nstudent = Student()\nstudent.cuda()\nwith open('teacher.params', 'rb') as f:\n    teacher = torch.load(f)\n\noptimizer = optim.Adam(student.parameters(), lr=lr)\n\ntrain_set = datasets.MNIST('../data', train=True, download=True,\n                       transform=transforms.Compose([\n                           transforms.ToTensor(),\n                           transforms.Normalize((0.1307,), (0.3081,))\n                       ]))\n\ntransfer_data, transfer_labels = [], []\nfor data, label in train_set:\n    if label != 3:\n        transfer_data.append(data.tolist())\n        transfer_labels.append(label)\n        \ntransfer_data, transfer_labels = torch.Tensor(transfer_data), torch.Tensor(transfer_labels)\n\nkwargs = {'num_workers': 1, 'pin_memory': True}\ntransfer_loader = torch.utils.data.DataLoader(\n        torch.utils.data.TensorDataset(transfer_data, transfer_labels),\n        batch_size=batch_size, shuffle=True, **kwargs)\n\ntest_loader = torch.utils.data.DataLoader(\n        datasets.MNIST('../data', train=False, transform=transforms.Compose([\n                    transforms.ToTensor(),\n                    transforms.Normalize((0.1307,), (0.3081,))\n                ])),\n        batch_size=batch_size, shuffle=True, **kwargs)\n\nprint(teacher)\n\nteacher.eval()\ntrain()\ntest()")

