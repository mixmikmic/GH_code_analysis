import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

torch.manual_seed(42)

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

class DCNet(nn.Module):

    def __init__(self, hidden_dim, layer1_dim, layer2_dim):
        super(DCNet, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(4, hidden_dim) 
        self.linear1 = nn.Linear(hidden_dim, layer1_dim)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(layer1_dim, layer2_dim)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(layer2_dim, 4)
        self.hidden_init_values = None
        self.hidden = self.init_hidden()
        nn.init.xavier_uniform(self.linear1.weight)
        nn.init.xavier_uniform(self.linear2.weight)
        nn.init.xavier_uniform(self.linear3.weight)
        
    def init_hidden(self):
        if self.hidden_init_values == None:
            self.hidden_init_values = (autograd.Variable(torch.randn(1, 1, self.hidden_dim)),
                                       autograd.Variable(torch.randn(1, 1, self.hidden_dim)))
        return self.hidden_init_values

    def forward(self, seq):
        lstm_out, self.hidden = self.lstm(
            seq.view(len(seq), 1, -1), self.hidden)
        tmp1 = self.relu1(self.linear1(lstm_out.view(len(seq), -1)))
        tmp2 = self.relu2(self.linear2(tmp1))
        _out = self.linear3(tmp2)
        base_out = _out
        return base_out

import random
torch.set_default_tensor_type('torch.cuda.FloatTensor')

bmap = {"A":0, "C":1, "G":2, "T":3}
def one_hot(b):
    t = [[0,0,0,0]]
    i = bmap[b]
    t[0][i] = 1
    return t

print("one-hot encoding for DNA bases")
print("A:", one_hot("A"))
print("C:", one_hot("C"))
print("G:", one_hot("G"))
print("T:", one_hot("T"))

def sim_error(seq, pi=0.05, pd=0.05, ps=0.01):
    """
    Given an input sequence `seq`, generating another
    sequence with errors. 
    pi: insertion error rate
    pd: deletion error rate
    ps: substitution error rate
    """
    out_seq = []
    for c in seq:
        while 1:
            r = random.uniform(0,1)
            if r < pi:
                out_seq.append(random.choice(["A","C","G","T"]))
            else:
                break
        r -= pi
        if r < pd:
            continue
        r -= pd
        if r < ps:
            out_seq.append(random.choice(["A","C","G","T"]))
            continue
        out_seq.append(c)
    return "".join(out_seq)

seq1 = list("AAAAAAA")+[random.choice(["A","C","G","T"]) for _ in range(220)]
print("seq1", "".join(seq1))
# convert the `seq` to a PyTorch tensor
print()
seq2 = list("TTTTTTT")+[random.choice(["A","C","G","T"]) for _ in range(220)]
print("seq2", "".join(seq2))
# convert the `seq` to a PyTorch tensor

seq1_t = Variable(torch.FloatTensor([one_hot(c) for c in seq1])).cuda()
seq2_t = Variable(torch.FloatTensor([one_hot(c) for c in seq2])).cuda()

seqs1 = [sim_error(seq1, pi=0.05, pd=0.05, ps=0.01) for _ in range(30)]
seqs2 = [sim_error(seq2, pi=0.05, pd=0.05, ps=0.01) for _ in range(30)]
seqs_all_t= [Variable(torch.FloatTensor([one_hot(c) for c in s])).cuda() for s in seqs1+seqs2]

dcnet = DCNet(32, 12, 12)
dcnet.cuda()
dcnet.zero_grad()
dcnet.hidden = dcnet.init_hidden()

# initial the paramerters in the DCNet
for name, param in dcnet.named_parameters():
    if 'bias' in name:
        nn.init.constant(param, 0.0)
    elif 'weight' in name:
        nn.init.xavier_normal(param)

#loss_function = nn.L1Loss()
loss_function = nn.MSELoss()
lr = 0.1
optimizer = optim.SGD(dcnet.parameters(), lr=lr)

range_ = (1, 200)
mini_batch_size = 5
for epoch in range(4501):
    for i in range(int(len(seqs_all_t)/mini_batch_size)):
        loss = 0
        s, e = range_
        for tmp_seq in random.sample(seqs_all_t, mini_batch_size):
            dcnet.hidden = dcnet.init_hidden()
            dcnet.zero_grad()
            tmp_seq = tmp_seq[s-1:e]
            seq_ = tmp_seq.view(-1,4)
            out = dcnet(seq_)
            loss += loss_function(out[:-1], seq_[1:])
        loss.backward()
        optimizer.step()
    if epoch % 250==0:
        print("epoch:", epoch, "loss:", loss.cpu().data[0]/mini_batch_size, "learning rate:", lr)
        lr *= 0.85
        optimizer = optim.SGD(dcnet.parameters(), lr=lr)

import numpy as np
dcnet.hidden = dcnet.init_hidden()
xout = dcnet(seq1_t[:250])
x1 = xout[:-1].cpu().data.numpy() 
xx1=np.transpose(seq1_t[1:250,0,:].data.cpu().numpy())
xx2=np.transpose(x1)

plt.figure(figsize=(18,3))
plt.subplot(3,1,1)
plt.matshow(xx1, vmin=-0.1, vmax=1.1, fignum=False)
plt.text(0,6,"the tensor for the original template")
frame = plt.gca()
frame.axes.yaxis.set_ticklabels([])

plt.subplot(3,1,2)
plt.matshow(xx2, vmin=-0.1, vmax=1.1, fignum=False)
plt.text(0,6,"reconstructed tensor from the DCNet")
frame = plt.gca()
frame.axes.yaxis.set_ticklabels([])

plt.subplot(3,1,3)
plt.matshow(xx1-xx2, vmin=-0.1, vmax=1.1, fignum=False)
plt.text(0,6,"differences");
frame = plt.gca()
frame.axes.yaxis.set_ticklabels([]);

dcnet.hidden = dcnet.init_hidden()
xout = dcnet(seq2_t[:250])
x1 = xout[:-1].cpu().data.numpy() 
xx1=np.transpose(seq2_t[1:250,0,:].data.cpu().numpy())
xx2=np.transpose(x1)

plt.figure(figsize=(18,3))
plt.subplot(3,1,1)
plt.matshow(xx1, vmin=-0.1, vmax=1.1, fignum=False)
plt.text(0,6,"the tensor for the original template")
frame = plt.gca()
frame.axes.yaxis.set_ticklabels([])

plt.subplot(3,1,2)
plt.matshow(xx2, vmin=-0.1, vmax=1.1, fignum=False)
plt.text(0,6,"reconstructed tensor from the DCNet")
frame = plt.gca()
frame.axes.yaxis.set_ticklabels([])

plt.subplot(3,1,3)
plt.matshow(xx1-xx2, vmin=-0.1, vmax=1.1, fignum=False)
plt.text(0,6,"differences");
frame = plt.gca()
frame.axes.yaxis.set_ticklabels([]);

dcnet.hidden = dcnet.init_hidden()
base_t= Variable(torch.FloatTensor([one_hot(c) for c in seq1[:5]])).cuda()

consensus = []

for _ in range(201):
    xout = dcnet(base_t)
    next_t = [0,0,0,0]
    next_t[np.argmax(xout.cpu().data.numpy()[-1])]=1
    consensus.append(next_t)
    base_t= Variable(torch.FloatTensor([next_t])).cuda()
    
consensus = np.array(consensus)
consensus = consensus.transpose()

xx1=np.transpose(seq1_t[5:250,0,:].data.cpu().numpy())

plt.figure(figsize=(18,3))
plt.subplot(3,1,1)
plt.matshow(xx1[:,0:201], vmin=-0.1, vmax=1.1, fignum=False)
plt.text(0,6,"the tensor for the original template")
frame = plt.gca()
frame.axes.yaxis.set_ticklabels([])

plt.subplot(3,1,2)
plt.matshow(consensus, vmin=-0.1, vmax=1.1, fignum=False)
plt.text(0,6,"generated consensus from the DCNet")
frame = plt.gca()
frame.axes.yaxis.set_ticklabels([])

plt.subplot(3,1,3)
plt.matshow(consensus-xx1[:,0:201], vmin=-0.1, vmax=1.1, fignum=False)
plt.text(0,6,"differences")
frame = plt.gca()
frame.axes.yaxis.set_ticklabels([]);

dcnet.hidden = dcnet.init_hidden()
base_t= Variable(torch.FloatTensor([one_hot(c) for c in seq2[:5]])).cuda()

consensus = []

for _ in range(201):
    xout = dcnet(base_t)
    next_t = [0,0,0,0]
    next_t[np.argmax(xout.cpu().data.numpy()[-1])]=1
    consensus.append(next_t)
    base_t= Variable(torch.FloatTensor([next_t])).cuda()
    
consensus = np.array(consensus)
consensus = consensus.transpose()


xx1=np.transpose(seq2_t[5:250,0,:].data.cpu().numpy())

plt.figure(figsize=(18,3))
plt.subplot(3,1,1)
plt.matshow(xx1[:,0:201], vmin=-0.1, vmax=1.1, fignum=False)
plt.text(0,6,"the tensor for the original template")
frame = plt.gca()
frame.axes.yaxis.set_ticklabels([])

plt.subplot(3,1,2)
plt.matshow(consensus, vmin=-0.1, vmax=1.1, fignum=False)
plt.text(0,6,"generated consensus from the DCNet")
frame = plt.gca()
frame.axes.yaxis.set_ticklabels([])

plt.subplot(3,1,3)
plt.matshow(consensus-xx1[:,0:201], vmin=-0.1, vmax=1.1, fignum=False)
plt.text(0,6,"differences")
frame = plt.gca()
frame.axes.yaxis.set_ticklabels([]);

