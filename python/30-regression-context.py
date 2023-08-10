import matplotlib as mpl
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('bmh')

import numpy as np

import ujson
import attr
import random
import torch

from itertools import islice
from tqdm import tqdm_notebook
from glob import glob
from collections import Counter

from gensim.models import KeyedVectors

from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

vectors = KeyedVectors.load_word2vec_format(
    '../data/vectors/GoogleNews-vectors-negative300.bin.gz',
    binary=True,
)

class Corpus:
    
    def __init__(self, pattern, skim=None):
        self.pattern = pattern
        self.skim = skim
        
    def lines(self):
        for path in glob(self.pattern):
            with open(path) as fh:
                for line in fh:
                    yield line.strip()

    def abstracts(self):
        lines = self.lines()
        if self.skim:
            lines = islice(lines, self.skim)
        for line in tqdm_notebook(lines, total=self.skim):
            raw = ujson.loads(line)
            yield Abstract.from_raw(raw)
            
    def xy(self):
        for abstract in self.abstracts():
            yield from abstract.xy()

@attr.s
class Abstract:
    
    sentences = attr.ib()
    
    @classmethod
    def from_raw(cls, raw):
        return cls([Sentence(s['token']) for s in raw['sentences']])
            
    def xy(self):
        for i, sent in enumerate(self.sentences):
            try:
                x = sent.tensor()
                y = i / (len(self.sentences)-1)
                y = torch.FloatTensor([y])
                yield x, y
            except RuntimeError as e:
                pass

@attr.s
class Sentence:
    
    tokens = attr.ib()
    
    def tensor(self, dim=300, pad=50):
        x = [vectors[t] for t in self.tokens if t in vectors]
        x += [np.zeros(dim)] * pad
        x = x[:pad]
        x = np.array(x)
        x = torch.from_numpy(x)
        x = x.float()
        return x

train = Corpus('../data/train.json/*.json', 100)

class SentenceEncoder(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(embed_dim, hidden_dim, nonlinearity='relu', batch_first=True)

    def forward(self, x):
        x = Variable(x.unsqueeze(0))
        hidden = Variable(torch.zeros(1, 1, self.hidden_dim))
        _, hidden = self.rnn(x, hidden)
        return hidden

class ContextEncoder(nn.Module):
    
    def __init__(self, sent_encoder, hidden_dim):
        super().__init__()
        self.sent_encoder = sent_encoder
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(hidden_dim, hidden_dim, nonlinearity='relu', batch_first=True)

    def forward(self, x):
        x = torch.cat([self.sent_encoder(s) for s in x], 1)
        hidden = Variable(torch.zeros(1, 1, self.hidden_dim))
        _, hidden = self.rnn(x, hidden)
        return hidden

class Regressor(nn.Module):
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden2y = nn.Linear(2*hidden_dim, 1)
        
    def forward(self, x):
        return self.hidden2y(x)

torch.manual_seed(1)

sent_encoder = SentenceEncoder(300, 150)

ctx_encoder = ContextEncoder(sent_encoder, 150)

regressor = Regressor(150)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_loss = []
for epoch in range(5):
    
    print(f'Epoch {epoch}')
    
    epoch_loss = 0
    for ab in train.abstracts():

        xy = list(ab.xy())
        random.shuffle(xy)

        ctx, _ = zip(*xy)
        ctx = ctx_encoder(ctx)

        for sent, y in xy:

            sent = sent_encoder(sent)
            x = torch.cat([ctx, sent], 2)        
            y = Variable(y)

            y_pred = regressor(x)

            loss = criterion(y_pred, y)
            loss.backward(retain_graph=True)

            optimizer.step()
        
        epoch_loss += loss.data[0]
        
    epoch_loss /= train.skim
    train_loss.append(epoch_loss)
    print(epoch_loss)



