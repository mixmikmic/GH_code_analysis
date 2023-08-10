import numpy as np
import torch

# Create a 3 x 2 array
np.ndarray((3, 2))

# Create a 3 x 2 Tensor
torch.Tensor(3, 2)

a = torch.Tensor([1,2])
b = torch.Tensor([3,4])
print('a + b:', a + b)
print('a - b:', a - b)
print('a * b:', a * b)
print('a / b:', a / b)

a = torch.Tensor(5, 5)
print('a:', a)

# Slice using ranges
print('a[2:4, 3:4]', a[2:4, 3:4])

# Can count backwards using negative indices
print('a[:, -1]', a[:, -1])

# Skipping elements
print('a[::2, ::3]', a[::2, ::3])

# Tensor from array
arr = np.array([1,2])
torch.from_numpy(arr)

# Tensor to array
t = torch.Tensor([1, 2])
t.numpy()

t = torch.Tensor([1, 2]) # on CPU
if torch.cuda.is_available():
    t = t.cuda() # on GPU

from torch.autograd import Variable

# Data
x = Variable(torch.Tensor([1,  2,  3,  4]))
y = Variable(torch.Tensor([0, -1, -2, -3]))

# Initialize a variables
m = Variable(torch.rand(1), requires_grad=True)
b = Variable(torch.rand(1), requires_grad=True)

# Define function
y_hat = m * x + b

# Define loss
loss = torch.mean(0.5 * (y - y_hat)**2)

loss.backward() # Backprop the gradients of the loss w.r.t other variables

# Gradients
print('dL/dm: %0.4f' % m.grad.data)
print('dL/db: %0.4f' % b.grad.data)

import torch.nn as nn

class LinearModel(nn.Module):
    
    def __init__(self):
        """This method is called when you instantiate a new LinearModel object.
        
        You should use it to define the parameters/layers of your model.
        """
        # Whenever you define a new nn.Module you should start the __init__()
        # method with the following line. Remember to replace `LinearModel` 
        # with whatever you are calling your model.
        super(LinearModel, self).__init__()
        
        # Now we define the parameters used by the model.
        self.m = torch.nn.Parameter(torch.rand(1))
        self.b = torch.nn.Parameter(torch.rand(1))
    
    def forward(self, x):
        """This method computes the output of the model.
        
        Args:
            x: The input data.
        """
        return self.m * x + self.b


# Example forward pass. Note that we use model(x) not model.forward(x) !!! 
model = LinearModel()
y_hat = model(x)

import torch.optim as optim
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

import time

for i in range(5001):
    optimizer.zero_grad()
    y_hat = model(x)
    loss = torch.mean(0.5 * (y - y_hat)**2)
    loss.backward()
    optimizer.step()
    if i % 1000 == 0:
        time.sleep(1) # DO NOT INCLUDE THIS IN YOUR CODE !!! Only for demo.
        print('Iteration %i - Loss: %0.6f' % (i, loss.data[0]),
              end='\r') # COOL TRICK: ` end='\r' ` makes print overwrite the current line.

print('Final parameters:')
print('m: %0.2f' % model.m.data[0])
print('b: %0.2f' % model.b.data[0])

get_ipython().system('wget http://mattmahoney.net/dc/text8.zip')
get_ipython().system('unzip text8.zip')

with open('text8', 'r') as f:
    corpus = f.read().split()

from collections import Counter


def build_vocabulary(corpus):
    """Builds a vocabulary.
    
    Args:
        corpus: A list of words.
    """
    counts = Counter(corpus) # Count the word occurances.
    counts = counts.items() # Transform Counter to (word, count) tuples.
    counts = sorted(counts, key=lambda x: x[1], reverse=True) # Sort in terms of frequency.
    reverse_vocab = [x[0] for x in counts] # Use a list to map indices to words.
    vocab = {x: i for i, x in enumerate(reverse_vocab)} # Invert that mapping to get the vocabulary.
    data = [vocab[x] for x in corpus] # Get ids for all words in the corpus.
    return data, vocab, reverse_vocab


data, vocab, reverse_vocab = build_vocabulary(corpus)

reverse_vocab[data[0]]

from collections import deque
from random import shuffle
from torch.autograd import Variable


def batch_generator(data, batch_size, window_size):

    # Stores the indices of the words in order that they will be chosen when generating batches.
    # e.g. the first word that will be chosen in an epoch is word_ids[0]
    ids = list(range(window_size, len(data) - window_size))

    while True:
        shuffle(ids) # Randomize at the start of each epoch
        sample_queue = deque()
        for id in ids: # Iterate over random words
            w = data[id]
            c = []
            for i in range(1, window_size + 1): # Iterate over window sizes
                c.append(data[id - i]) # Left context
                c.append(data[id + i]) # Right context
            sample_queue.append((w, c))
            
            # Once positive sample queue is full deque a batch and generate negative samples
            if len(sample_queue) >= batch_size:
                
                batch = [sample_queue.pop() for _ in range(batch_size)]
                w, c = zip(*batch) # Separate words and contexts
                
                # Read data into torch variables
                w = Variable(torch.LongTensor(w))
                c = Variable(torch.LongTensor(c))
                
                # Transfer data to GPU if available
                if torch.cuda.is_available():
                    w = w.cuda()
                    c = c.cuda()
                    
                yield w, c

# Training hyperparameters
batch_size = 4
window_size = 1

# Example usage
it = batch_generator(data, batch_size, window_size) # Initialize the iterable
w, c = next(it) # The next() function will run until yield
print('w:', w)
print('c:', c)

import torch.nn as nn
import torch.nn.functional as F


class CBOW(nn.Module):
    
    def __init__(self, vocab_size, embedding_size):
        super(CBOW, self).__init__()
        # Parameters
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        # Layers
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.fc = nn.Linear(embedding_size, vocab_size)
        # Initialize embedding weights
        initrange = 0.5 / embedding_size
        self.embeddings.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, c):
        batch_size, window_size = c.size()
        net = c.view(-1) # Flatten window dimension
        net = self.embeddings(net) # Get context embeddings
        net = net.view(batch_size, window_size, self.embedding_size) # Unflatten window dimension
        net = torch.sum(net, dim=1) # Sum over window dimension
        net = self.fc(net) # Feed into fully-connected layer
        net = F.log_softmax(net, dim=1) # Log probabilities
        return net

from torch import optim

# Training settings / hyperparameters
n_iterations = 10000
batch_size = 256
window_size = 10
it = batch_generator(data, batch_size, window_size)

vocab_size = len(vocab)
embedding_size = 128
model = CBOW(vocab_size, embedding_size)
if torch.cuda.is_available():
    model = model.cuda()

loss_function = nn.NLLLoss()
# optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters())

for i in range(n_iterations):
    w, c = next(it) # Generate a new batch of data
    optimizer.zero_grad()
    output = model(c)
    loss = loss_function(output, w)
    loss.backward()
    optimizer.step()
    if (i % 100) == 0:
        print('Iteration %i - Loss: %0.6f' % (i, loss.data[0]), end='\r')

torch.save(model, 'model.pt')

model = torch.load('model.pt')

# Extract the embedding weights from the model
embeddings = model.embeddings.weight.data

# Transform the weight tensor to a numpy array
if torch.cuda.is_available():
    embeddings = embeddings.cpu().numpy()
else:
    embeddings = embeddings.numpy()

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
get_ipython().magic('matplotlib inline')

# Get the t-SNE embeddings.
tsne = TSNE(n_components=2).fit_transform(embeddings[:2000])

# Create a scatter plot
fig = plt.figure(figsize=(12, 12), dpi=900)
ax = fig.add_subplot(111)
ax.scatter(tsne[:,0], tsne[:,1])

# Randomly select some words and label them
random_choices = np.random.choice(2000, 100, replace=False)
for i in random_choices:
    ax.text(tsne[i, 0], tsne[i, 1], reverse_vocab[i], fontsize=12, color='orange')

fig.show()

import scipy

def similar_words(word,
                  embeddings=embeddings,
                  vocab=vocab,
                  reverse_vocab=reverse_vocab):
    print('Most similar words to: %s' % word)
    # Get embedding for query word.
    word_idx = vocab[word]
    input_embedding = embeddings[word_idx:word_idx+1, :]
    # Compute cosine distance between query word embedding and all other embeddings.
    cosine_distance = scipy.spatial.distance.cdist(input_embedding, embeddings, 'cosine')
    # Find closest embeddings and print out corresponding words.
    closest_word_ids = np.argsort(cosine_distance)[0][1:11]
    for i, close_word_idx in enumerate(closest_word_ids):
        print('%i - %s' % (i + 1, reverse_vocab[close_word_idx]))

similar_words('food')

