import numpy as np
from collections import Counter

history_length = 3

def process_input(filename):
    rawtext = open(filename).read()
    sequences = [rawtext[i:i+history_length] for i in range(int(len(rawtext)/history_length))]
    stats = Counter(sequences)
    tokens = []
    counts = []
    for i in stats.most_common():
        tokens.append(i[0])
        counts.append(i[1])
    return stats
    
def next_char(cur,stats):
    seed = cur[1:]
    candidates = []
    candidatec = []
    for k in stats.keys():
        if seed==k[:-1]:
            candidates.append(k)
            candidatec.append(float(stats[k]))
    candidatep = [x/sum(candidatec) for x in candidatec]
    return candidates[np.random.choice(len(candidatec),p=candidatep)]

def sample(length,running_state, stats):
    output = ''
    for i in range(length):
        output+=running_state[0]
        running_state = next_char(running_state,stats)
    return output

tweet_stats = process_input('tweets.txt')
sample(50,'oba',tweet_stats)

hamlet_stats = process_input('hamlet.txt')
sample(50, 'the',hamlet_stats)

history_length = 4
tweet_stats = process_input('tweets.txt')
print(sample(50,'and ',tweet_stats))
hamlet_stats = process_input('hamlet.txt')
print(sample(50,'the ',hamlet_stats))

history_length = 5
tweet_stats = process_input('tweets.txt')
print(sample(50,'obama',tweet_stats))
hamlet_stats = process_input('hamlet.txt')
print(sample(50,'to be',hamlet_stats))

# This code is meant as an overview, not meant to compile
'''
def markov_generator(x):
    current_state = current_state
    for i in range(some target length):
        output += generate_single_step_output(current_state)
        new_state = random_process(current_state)
        current_state = new_state
    return output
'''

# This code is meant as an overview, not meant to compile
'''
class generator_with_memory:
    def __init__():
        self.memory_state = self.init_memory()
    def step(x):
        self.memory_state = smart_process(x,self.memory_state)
        y = generate_output(self.memory_state)
        return y
'''

# This code is meant as an overview, not meant to compile
'''
def step(x):
    self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
    y = np.dot(self.W_hy, self.h)
    return y
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)

T = 80    # range from which ints are sampled
L = 1000  # Length of generated sequence
N = 100   # number of examples
future = 1000 # length of sequence to predict
# generating a sinusoidal time series
x = np.empty((N, L), 'int64')
x[:] = np.array(range(L)) + np.random.randint(- 4 * T, 4 * T, N).reshape(N, 1)
data = np.sin(x / 1.0 / T).astype('float64')


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 32)
        self.lstm2 = nn.LSTMCell(32, 32)
        self.linear = nn.Linear(32, 1)
    def forward(self, input, future = 0):
        outputs = []
        h_t = Variable(torch.zeros(input.size(0), 32).double(), requires_grad=False)
        c_t = Variable(torch.zeros(input.size(0), 32).double(), requires_grad=False)
        h_t2 = Variable(torch.zeros(input.size(0), 32).double(), requires_grad=False)
        c_t2 = Variable(torch.zeros(input.size(0), 32).double(), requires_grad=False)
        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):# predicting future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs
    
def save_plot_wave(y_gen):
    plt.figure(figsize=(30,10))
    plt.title('Predict future values for time sequence', fontsize=30)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(np.arange(input.size(1)), y_gen[0][:input.size(1)], 'b', linewidth = 2.0)
    plt.plot(np.arange(input.size(1), input.size(1) + future), y_gen[0][input.size(1):], 'b' + ':', linewidth = 2.0)
    plt.savefig('predict%d.pdf'%i)
    plt.close()

input = Variable(torch.from_numpy(data[1:, :-1]), requires_grad=False)
target = Variable(torch.from_numpy(data[1:, 1:]), requires_grad=False)
test_input = Variable(torch.from_numpy(data[:1, :-1]), requires_grad=False)
test_target = Variable(torch.from_numpy(data[:1, 1:]), requires_grad=False)

# build the model
seq = Sequence()
seq.double()
criterion = nn.MSELoss()
optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
#begin to train
for i in range(11):
    print('Step: ', i)
    def closure():
        optimizer.zero_grad()
        out = seq(input)
        loss = criterion(out, target)
        print('Loss:', loss.data.numpy()[0])
        loss.backward()
        return loss
    optimizer.step(closure)
    # begin to predict
    pred = seq(test_input, future = future)
    loss = criterion(pred[:, :-future], test_target)
    print('Test loss:', loss.data.numpy()[0])
    y = pred.data.numpy()
    save_plot_wave(y)



