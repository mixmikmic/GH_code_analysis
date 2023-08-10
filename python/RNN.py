from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe

TEXT = data.Field(lower=True)

train, valid, test = datasets.WikiText2.splits(TEXT)

TEXT.build_vocab(train, vectors=GloVe(name="6B", dim=EMBEDDING_DIM))

train_iter, valid_iter, test_iter = data.BPTTIterator.splits(
    (train, valid, test), batch_size=BATCH_SIZE, bptt_len=BPTT_LEN, repeat=False,
    device=-1)

TEXT.build_vocab(train, vectors=GloVe(name="6B", dim=EMBEDDING_DIM))

TEXT.vocab.freqs.most_common(5)

train_iter, valid_iter, test_iter = data.BPTTIterator.splits(
    (train, valid, test), batch_size=BATCH_SIZE, bptt_len=BPTT_LEN, repeat=False,
    device=-1)

batch = next(iter(train_iter))
data = batch.text.transpose(1, 0).data.numpy()
sample = []
for d1 in data:
    for d2 in d1:
        sample.append(TEXT.vocab.itos[d2])
print(" ".join(sample))

print("Total Training Data:", len(train_iter))
print("Total Validation Data:", len(train_iter))
print("Total Testing Data:", len(train_iter))
print("Total Vocabularies:", len(TEXT.vocab))

import logging
 
# add filemode="w" to overwrite
logging.basicConfig(filename="sample.log", level=logging.INFO)

import torch
import torch.nn as nn
from torch.optim import Adam

############################
# Variable Initialization #
############################
BATCH_SIZE = 64
BPTT_LEN = 30
EMBEDDING_DIM = 300
HIDDEN_SIZE = 512
NUM_LAYERS = 1
DROPOUT = 0.5
VOCAB_SIZE = len(TEXT.vocab)

#################################
# Neural Network Initialization #
#################################
class LanguageModelLSTM(nn.Module):
    def __init__(self):
        super(LanguageModelLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=EMBEDDING_DIM,
                            hidden_size=HIDDEN_SIZE,
                            num_layers=NUM_LAYERS,
                            dropout=DROPOUT)
        self.linear = nn.Linear(in_features=HIDDEN_SIZE,
                                out_features=VOCAB_SIZE)
        
    def forward(self, X):
        lstm_out, lstm_hidden = self.lstm(X)
        step_size, batch_size, _ = lstm_out.size()
        modified_output = lstm_out.view(step_size * batch_size, -1)
        
        out = self.linear(modified_output)
        
        return out
    
embedding = nn.Embedding(TEXT.vocab.vectors.size(0),
                         TEXT.vocab.vectors.size(1))
embedding.weight.data.copy_(TEXT.vocab.vectors)
model = LanguageModelLSTM()
loss_fn = nn.CrossEntropyLoss()
opt = Adam(model.parameters())

if torch.cuda.is_available():
    embedding.cuda()
    model.cuda()
    loss_fn.cuda()
    
model.load_state_dict(torch.load("lm.pt"))

################
# RNN Training #
################
total_steps = len(train_iter)
for epoch in range(100):
    logging.info("Epoch %d..." % epoch)
    for idx, batch in enumerate(train_iter):
        model.zero_grad()
        if torch.cuda.is_available():
            inp = batch.text.cuda()
            trg = batch.target.cuda()
        else:
            inp = batch.text
            trg = batch.target
        word_embedding = embedding(inp)
        out = model(word_embedding)
        target = trg.view(-1)
        loss = loss_fn(out, target)

        if idx % 100 == 0:
            logging.info("Loss [%d/%d]: %f" % (idx, total_steps, loss.data.cpu().numpy()[0]))

        loss.backward()

        opt.step()

for i, batch in enumerate(test_iter):
    if torch.cuda.is_available():
        inp = batch.text.cuda()
    else:
        inp = batch.text
    word_embedding = embedding(inp)
    out = model(word_embedding)
    values, indices = out.max(1)
    
    print("PREDICTION: ")
    for idx in indices.data.cpu().numpy():
        print(TEXT.vocab.itos[idx], end=" ")
    print("\n\nREAL LABEL: ")
    for idx in batch.text.transpose(1, 0).data.numpy():
        for idx2 in idx:
            print(TEXT.vocab.itos[idx2], end=" ")
            
    break

