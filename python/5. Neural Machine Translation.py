import time
import numpy as np
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

import numpy as np
import codecs
import nltk

train_lines = [line.strip().split('\t') for line in codecs.open('data/jpn-train.txt', 'r', encoding='utf-8')]
dev_lines = [line.strip().split('\t') for line in codecs.open('data/jpn-dev.txt', 'r', encoding='utf-8')]
test_lines = [line.strip().split('\t') for line in codecs.open('data/jpn-test.txt', 'r', encoding='utf-8')]

src_vocab = set()
trg_vocab = set()
for line in train_lines:
    for word in line[1]:
        if word not in src_vocab:
            src_vocab.add(word)
    for word in line[0].split():
        if word not in trg_vocab:
            trg_vocab.add(word)

# Add special tokens to the source and target vocabularies
src_vocab.add('<s>')
src_vocab.add('</s>')
src_vocab.add('<unk>')
src_vocab.add('<pad>')

trg_vocab.add('<s>')
trg_vocab.add('</s>')
trg_vocab.add('<unk>')
trg_vocab.add('<pad>')

src_word2id = {word: idx for idx, word in enumerate(src_vocab)}
src_id2word = {idx: word for idx, word in enumerate(src_vocab)}

trg_word2id = {word: idx for idx, word in enumerate(trg_vocab)}
trg_id2word = {idx: word for idx, word in enumerate(trg_vocab)}

print('Number of unique Japanese words : %d ' % (len(src_vocab)))
print('Number of unique English words : %d ' % (len(trg_vocab)))



class Seq2Seq(nn.Module):
    """A Vanilla Sequence to Sequence (Seq2Seq) model with LSTMs.
    Ref: Sequence to Sequence Learning with Neural Nets
    https://arxiv.org/abs/1409.3215
    """

    def __init__(
        self, src_emb_dim, trg_emb_dim, src_vocab_size,
        trg_vocab_size, src_hidden_dim, trg_hidden_dim,
        pad_token_src, pad_token_trg, bidirectional=False,
        nlayers_src=1, nlayers_trg=1
    ):
        """Initialize Seq2Seq Model."""
        super(Seq2Seq, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.src_emb_dim = src_emb_dim
        self.trg_emb_dim = trg_emb_dim
        self.src_hidden_dim = src_hidden_dim
        self.trg_hidden_dim = trg_hidden_dim
        self.bidirectional = bidirectional
        self.nlayers_src = nlayers_src
        self.nlayers_trg = nlayers_trg
        self.pad_token_src = pad_token_src
        self.pad_token_trg = pad_token_trg
        
        # Word Embedding look-up table for the soruce language
        self.src_embedding = nn.Embedding(
            self.src_vocab_size,
            self.src_emb_dim,
            self.pad_token_src,
        )

        # Word Embedding look-up table for the target language
        self.trg_embedding = nn.Embedding(
            self.trg_vocab_size,
            self.trg_emb_dim,
            self.pad_token_trg,
        )

        # Encoder GRU
        self.encoder = nn.GRU(
            self.src_emb_dim // 2 if self.bidirectional else self.src_emb_dim,
            self.src_hidden_dim,
            self.nlayers_src,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Decoder GRU
        self.decoder = nn.GRU(
            self.trg_emb_dim,
            self.trg_hidden_dim,
            self.nlayers_trg,
            batch_first=True
        )
        
        # Projection layer from decoder hidden states to target language vocabulary
        self.decoder2vocab = nn.Linear(trg_hidden_dim, trg_vocab_size)

    def forward(self, input_src, input_trg, src_lengths):
        # Lookup word embeddings in source and target minibatch
        src_emb = self.src_embedding(input_src)
        trg_emb = self.trg_embedding(input_trg)
        
        # Pack padded sequence for length masking in encoder RNN (This requires sorting input sequence by length)
        src_emb = pack_padded_sequence(src_emb, src_lengths, batch_first=True)
        
        # Run sequence of embeddings through the encoder GRU
        _, src_h_t = self.encoder(src_emb)
        
        # Extract the last hidden state of the GRU
        h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1) if self.bidirectional else src_h_t[-1]

        # Initialize the decoder GRU with the last hidden state of the encoder and 
        # run target inputs through the decoder.
        trg_h, _ = self.decoder(trg_emb, h_t.unsqueeze(0).expand(self.nlayers_trg, h_t.size(0), h_t.size(1)))
        
        # Merge batch and time dimensions to pass to a linear layer
        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size(0) * trg_h.size(1), trg_h.size(2)
        )
        
        # Affine transformation of all decoder hidden states
        decoder2vocab = self.decoder2vocab(trg_h_reshape)
        
        # Reshape
        decoder2vocab = decoder2vocab.view(
            trg_h.size(0), trg_h.size(1), decoder2vocab.size(1)
        )

        return decoder2vocab
    
    def decode(self, decoder2vocab):
        # Turn decoder output into a probabiltiy distribution over vocabulary
        decoder2vocab_reshape = decoder2vocab.view(-1, decoder2vocab.size(2))
        word_probs = F.softmax(decoder2vocab_reshape)
        word_probs = word_probs.view(
            decoder2vocab.size(0), decoder2vocab.size(1), decoder2vocab.size(2)
        )

        return word_probs

def get_parallel_minibatch(lines, src_word2id, trg_word2id, index, batch_size, volatile=False):
        
        # Get source sentences for this minibatch
        src_lines = [
            ['<s>'] + list(line[1]) + ['</s>']
            for line in lines[index: index + batch_size]
        ]

        # Get target sentences for this minibatch
        trg_lines = [
            ['<s>'] + line[0].split() + ['</s>']
            for line in lines[index: index + batch_size]
        ]
        
        # Sort source sentences by length for length masking in RNNs
        src_lens = [len(line) for line in src_lines]
        sorted_indices = np.argsort(src_lens)[::-1]
        
        # Reorder sentences based on source lengths
        sorted_src_lines = [src_lines[idx] for idx in sorted_indices]
        sorted_trg_lines = [trg_lines[idx] for idx in sorted_indices]
        
        # Compute new sentence lengths
        sorted_src_lens = [len(line) for line in sorted_src_lines]
        sorted_trg_lens = [len(line) for line in sorted_trg_lines]
        
        # Get max source and target lengths to pad input and output sequences
        max_src_len = max(sorted_src_lens)
        max_trg_len = max(sorted_trg_lens)
        
        # Construct padded source input sequence
        input_lines_src = [
            [src_word2id[w] if w in src_word2id else src_word2id['<unk>'] for w in line] +
            [src_word2id['<pad>']] * (max_src_len - len(line))
            for line in sorted_src_lines
        ]

        # Construct padded target input sequence
        input_lines_trg = [
            [trg_word2id[w] if w in trg_word2id else trg_word2id['<unk>'] for w in line[:-1]] +
            [trg_word2id['<pad>']] * (max_trg_len - len(line))
            for line in sorted_trg_lines
        ]

        # Construct padded target output sequence (Note: Output sequence is just the input shifted by 1 position)
        # This is for teacher-forcing
        output_lines_trg = [
            [trg_word2id[w] if w in trg_word2id else trg_word2id['<unk>'] for w in line[1:]] +
            [trg_word2id['<pad>']] * (max_trg_len - len(line))
            for line in sorted_trg_lines
        ]

        input_lines_src = Variable(torch.LongTensor(input_lines_src), volatile=volatile)
        input_lines_trg = Variable(torch.LongTensor(input_lines_trg), volatile=volatile)
        output_lines_trg = Variable(torch.LongTensor(output_lines_trg), volatile=volatile)

        return {
            'input_src': input_lines_src,
            'input_trg': input_lines_trg,
            'output_trg': output_lines_trg,
            'src_lens': sorted_src_lens
        }

cuda_available = torch.cuda.is_available()

seq2seq = Seq2Seq(
    src_emb_dim=128, trg_emb_dim=128,
    src_vocab_size=len(src_word2id), trg_vocab_size=len(trg_word2id),
    src_hidden_dim=512, trg_hidden_dim=512,
    pad_token_src=src_word2id['<pad>'],
    pad_token_trg=trg_word2id['<pad>'],
)

if cuda_available:
    seq2seq = seq2seq.cuda()

optimizer = optim.Adam(seq2seq.parameters(), lr=4e-4)
weight_mask = torch.ones(len(trg_word2id))
if cuda_available:
    weight_mask = weight_mask.cuda()
weight_mask[trg_word2id['<pad>']] = 0
loss_criterion = nn.CrossEntropyLoss(weight=weight_mask)
batch_size = 64

for epoch in range(15):
    losses = []
    for j in range(0, len(train_lines), batch_size):
        # Get minibatch of examples
        minibatch = get_parallel_minibatch(
            lines=train_lines, src_word2id=src_word2id,
            trg_word2id=trg_word2id, index=j, batch_size=batch_size
        )
        
        if cuda_available:
            minibatch['input_src'] = minibatch['input_src'].cuda()
            minibatch['input_trg'] = minibatch['input_trg'].cuda()
            minibatch['output_trg'] = minibatch['output_trg'].cuda()
        
        decoder_out = seq2seq(
            input_src=minibatch['input_src'], input_trg=minibatch['input_trg'], src_lengths=minibatch['src_lens']
        )
        
        loss = loss_criterion(
            decoder_out.contiguous().view(-1, decoder_out.size(2)),
            minibatch['output_trg'].contiguous().view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to avoid exploding gradients
        torch.nn.utils.clip_grad_norm(seq2seq.parameters(), 5.)
        optimizer.step()
        losses.append(loss.data[0])
    
    dev_nll = []
    for j in range(0, len(dev_lines), batch_size):
        # Get minibatch of examples
        minibatch = get_parallel_minibatch(
            lines=dev_lines, src_word2id=src_word2id,
            trg_word2id=trg_word2id, index=j, batch_size=batch_size,
            volatile=True
        )
        
        if cuda_available:
            minibatch['input_src'] = minibatch['input_src'].cuda()
            minibatch['input_trg'] = minibatch['input_trg'].cuda()
            minibatch['output_trg'] = minibatch['output_trg'].cuda()
        
        decoder_out = seq2seq(
            input_src=minibatch['input_src'], input_trg=minibatch['input_trg'], src_lengths=minibatch['src_lens']
        )
        
        loss = loss_criterion(
            decoder_out.contiguous().view(-1, decoder_out.size(2)),
            minibatch['output_trg'].contiguous().view(-1)
        )

        dev_nll.append(loss.data[0])
    
    test_nll = []
    for j in range(0, len(test_lines), batch_size):
        # Get minibatch of examples
        minibatch = get_parallel_minibatch(
            lines=test_lines, src_word2id=src_word2id,
            trg_word2id=trg_word2id, index=j, batch_size=batch_size,
            volatile=True
        )
        
        if cuda_available:
            minibatch['input_src'] = minibatch['input_src'].cuda()
            minibatch['input_trg'] = minibatch['input_trg'].cuda()
            minibatch['output_trg'] = minibatch['output_trg'].cuda()
        
        decoder_out = seq2seq(
            input_src=minibatch['input_src'], input_trg=minibatch['input_trg'], src_lengths=minibatch['src_lens']
        )
        
        loss = loss_criterion(
            decoder_out.contiguous().view(-1, decoder_out.size(2)),
            minibatch['output_trg'].contiguous().view(-1)
        )

        test_nll.append(loss.data[0])
    
    print('Epoch : %d Training Loss : %.3f' % (epoch, np.mean(losses)))
    print('Epoch : %d Dev Loss : %.3f' % (epoch, np.mean(dev_nll)))
    print('Epoch : %d Test Loss : %.3f' % (epoch, np.mean(test_nll)))
    print('-------------------------------------------------------------')

# Get the first minibatch in the dev set.
minibatch = get_parallel_minibatch(
    lines=dev_lines, src_word2id=src_word2id,
    trg_word2id=trg_word2id, index=0, batch_size=batch_size,
    volatile=True
)

if cuda_available:
    minibatch['input_src'] = minibatch['input_src'].cuda()
    minibatch['input_trg'] = minibatch['input_trg'].cuda()
    minibatch['output_trg'] = minibatch['output_trg'].cuda()

# Run it through our model (in teacher forcing mode)
res = seq2seq(
    input_src=minibatch['input_src'], input_trg=minibatch['input_trg'], src_lengths=minibatch['src_lens']
)

# Pick the most likely word at each time step
res = res.data.cpu().numpy().argmax(axis=-1)

# Cast targets to numpy
gold = minibatch['output_trg'].data.cpu().numpy()

# Decode indices to words for predictions and gold
res = [[trg_id2word[x] for x in line] for line in res]
gold = [[trg_id2word[x] for x in line] for line in gold]

for r, g in zip(res, gold):
    if '</s>' in r:
        index = r.index('</s>')
    else:
        index = len(r)
    
    print('Prediction : %s ' % (' '.join(r[:index])))

    index = g.index('</s>')
    print('Gold : %s ' % (' '.join(g[:index])))
    print('---------------')



