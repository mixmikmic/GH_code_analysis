import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

vocab = set.union(set(training_data[0][0]), set(training_data[1][0]))
word_to_ix ={}
for word in vocab:
    if word not in word_to_ix:
        word_to_ix[word] = len(word_to_ix)
        
print(word_to_ix)
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

embedding_dim = 6
hidden_dim = 6

class POSTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(POSTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

model = POSTagger(embedding_dim, hidden_dim, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

print(model)

inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
print(tag_scores)

for epochs in range(100):
    for sentence, tags in training_data:
        inputs = prepare_sequence(sentence, word_to_ix)
        target = prepare_sequence(tags, tag_to_ix)
        
        model.zero_grad() # clear all the prev grads
        model.hidden = model.init_hidden() # clear the hidden layer
        
        out = model(inputs)
        loss = loss_function(out, target)
        
        loss.backward()
        optimizer.step()
        


inputs = prepare_sequence(training_data[0][0], word_to_ix)
print(model(inputs))

