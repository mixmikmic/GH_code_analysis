get_ipython().system('wget https://raw.githubusercontent.com/UniversalDependencies/UD_English/master/en-ud-dev.conllu')
get_ipython().system('wget https://raw.githubusercontent.com/UniversalDependencies/UD_English/master/en-ud-test.conllu')
get_ipython().system('wget https://raw.githubusercontent.com/UniversalDependencies/UD_English/master/en-ud-train.conllu')

from collections import Counter


class Vocab(object):
    def __init__(self, iter, max_size=None, sos_token=None, eos_token=None, unk_token=None):
        """Initialize the vocabulary.
        Args:
            iter: An iterable which produces sequences of tokens used to update
                the vocabulary.
            max_size: (Optional) Maximum number of tokens in the vocabulary.
            sos_token: (Optional) Token denoting the start of a sequence.
            eos_token: (Optional) Token denoting the end of a sequence.
            unk_token: (Optional) Token denoting an unknown element in a
                sequence.
        """
        self.max_size = max_size
        self.pad_token = '<pad>'
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        # Add special tokens.
        id2word = [self.pad_token]
        if sos_token is not None:
            id2word.append(self.sos_token)
        if eos_token is not None:
            id2word.append(self.eos_token)
        if unk_token is not None:
            id2word.append(self.unk_token)

        # Update counter with token counts.
        counter = Counter()
        for x in iter:
            counter.update(x)

        # Extract lookup tables.
        if max_size is not None:
            counts = counter.most_common(max_size)
        else:
            counts = counter.items()
            counts = sorted(counts, key=lambda x: x[1], reverse=True)
        words = [x[0] for x in counts]
        id2word.extend(words)
        word2id = {x: i for i, x in enumerate(id2word)}

        self._id2word = id2word
        self._word2id = word2id

    def __len__(self):
        return len(self._id2word)

    def word2id(self, word):
        """Map a word in the vocabulary to its unique integer id.
        Args:
            word: Word to lookup.
        Returns:
            id: The integer id of the word being looked up.
        """
        if word in self._word2id:
            return self._word2id[word]
        elif self.unk_token is not None:
            return self._word2id[self.unk_token]
        else:
            raise KeyError('Word "%s" not in vocabulary.' % word)

    def id2word(self, id):
        """Map an integer id to its corresponding word in the vocabulary.
        Args:
            id: Integer id of the word being looked up.
        Returns:
            word: The corresponding word.
        """
        return self._id2word[id]

import re
from torch.utils.data import Dataset


class Annotation(object):
    def __init__(self):
        """A helper object for storing annotation data."""
        self.tokens = []
        self.pos_tags = []


class CoNLLDataset(Dataset):
    def __init__(self, fname):
        """Initializes the CoNLLDataset.
        Args:
            fname: The .conllu file to load data from.
        """
        self.fname = fname
        self.annotations = self.process_conll_file(fname)
        self.token_vocab = Vocab([x.tokens for x in self.annotations],
                                 unk_token='<unk>')
        self.pos_vocab = Vocab([x.pos_tags for x in self.annotations])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        input = [self.token_vocab.word2id(x) for x in annotation.tokens]
        target = [self.pos_vocab.word2id(x) for x in annotation.pos_tags]
        return input, target

    def process_conll_file(self, fname):
        # Read the entire file.
        with open(fname, 'r') as f:
            raw_text = f.read()
        # Split into chunks on blank lines.
        chunks = re.split(r'^\n', raw_text, flags=re.MULTILINE)
        # Process each chunk into an annotation.
        annotations = []
        for chunk in chunks:
            annotation = Annotation()
            lines = chunk.split('\n')
            # Iterate over all lines in the chunk.
            for line in lines:
                # If line is empty ignore it.
                if len(line)==0:
                    continue
                # If line is a commend ignore it.
                if line[0] == '#':
                    continue
                # Otherwise split on tabs and retrieve the token and the
                # POS tag fields.
                fields = line.split('\t')
                annotation.tokens.append(fields[1])
                annotation.pos_tags.append(fields[3])
            if (len(annotation.tokens) > 0) and (len(annotation.pos_tags) > 0):
                annotations.append(annotation)
        return annotations

dataset = CoNLLDataset('en-ud-train.conllu')

input, target = dataset[0]
print('Example input: %s\n' % input)
print('Example target: %s\n' % target)
print('Translated input: %s\n' % ' '.join(dataset.token_vocab.id2word(x) for x in input))
print('Translated target: %s\n' % ' '.join(dataset.pos_vocab.id2word(x) for x in target))

import torch
from torch.autograd import Variable


def pad(sequences, max_length, pad_value=0):
    """Pads a list of sequences.
    Args:
        sequences: A list of sequences to be padded.
        max_length: The length to pad to.
        pad_value: The value used for padding.
    Returns:
        A list of padded sequences.
    """
    out = []
    for sequence in sequences:
        padded = sequence + [0]*(max_length - len(sequence))
        out.append(padded)
    return out


def collate_annotations(batch):
    """Function used to collate data returned by CoNLLDataset."""
    # Get inputs, targets, and lengths.
    inputs, targets = zip(*batch)
    lengths = [len(x) for x in inputs]
    # Sort by length.
    sort = sorted(zip(inputs, targets, lengths),
                  key=lambda x: x[2],
                  reverse=True)
    inputs, targets, lengths = zip(*sort)
    # Pad.
    max_length = max(lengths)
    inputs = pad(inputs, max_length)
    targets = pad(targets, max_length)
    # Transpose.
    inputs = list(map(list, zip(*inputs)))
    targets = list(map(list, zip(*targets)))
    # Convert to PyTorch variables.
    inputs = Variable(torch.LongTensor(inputs))
    targets = Variable(torch.LongTensor(targets))
    lengths = Variable(torch.LongTensor(lengths))
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        targets = targets.cuda()
        lengths = lengths.cuda()
    return inputs, targets, lengths

from torch.utils.data import DataLoader


for inputs, targets, lengths in DataLoader(dataset, batch_size=16, collate_fn=collate_annotations):
    print('Inputs: %s\n' % inputs.data)
    print('Targets: %s\n' % targets.data)
    print('Lengths: %s\n' % lengths.data)

    # Usually we'd keep sampling batches, but here we'll just break
    break

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Tagger(nn.Module):
    def __init__(self,
                 input_vocab_size,
                 output_vocab_size,
                 embedding_dim=64,
                 hidden_size=64,
                 bidirectional=True):
        """Initializes the tagger.
        
        Args:
            input_vocab_size: Size of the input vocabulary.
            output_vocab_size: Size of the output vocabulary.
            embedding_dim: Dimension of the word embeddings.
            hidden_size: Number of units in each LSTM hidden layer.
            bidirectional: Whether or not to use a bidirectional rnn.
        """
        # Always do this!!!
        super(Tagger, self).__init__()

        # Store parameters
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        
        # Define layers
        self.word_embeddings = nn.Embedding(input_vocab_size, embedding_dim,
                                            padding_idx=0)
        self.rnn = nn.GRU(embedding_dim, hidden_size,
                          bidirectional=bidirectional,
                          dropout=0.9)
        if bidirectional:
            self.fc = nn.Linear(2*hidden_size, output_vocab_size)
        else:
            self.fc = nn.Linear(hidden_size, output_vocab_size)
        self.activation = nn.LogSoftmax(dim=2)

    def forward(self, x, lengths=None, hidden=None):
        """Computes a forward pass of the language model.
        
        Args:
            x: A LongTensor w/ dimension [seq_len, batch_size].
            lengths: The lengths of the sequences in x.
            hidden: Hidden state to be fed into the lstm.
            
        Returns:
            net: Probability of the next word in the sequence.
            hidden: Hidden state of the lstm.
        """
        seq_len, batch_size = x.size()
        
        # If no hidden state is provided, then default to zeros.
        if hidden is None:
            if self.bidirectional:
                num_directions = 2
            else:
                num_directions = 1
            hidden = Variable(torch.zeros(num_directions, batch_size, self.hidden_size))
            if torch.cuda.is_available():
                hidden = hidden.cuda()

        net = self.word_embeddings(x)
        # Pack before feeding into the RNN.
        if lengths is not None:
            lengths = lengths.data.view(-1).tolist()
            net = pack_padded_sequence(net, lengths)
        net, hidden = self.rnn(net, hidden)
        # Unpack after
        if lengths is not None:
            net, _ = pad_packed_sequence(net)
        net = self.fc(net)
        net = self.activation(net)

        return net, hidden

import numpy as np

# Load datasets.
train_dataset = CoNLLDataset('en-ud-train.conllu')
dev_dataset = CoNLLDataset('en-ud-dev.conllu')

dev_dataset.token_vocab = train_dataset.token_vocab
dev_dataset.pos_vocab = train_dataset.pos_vocab

# Hyperparameters / constants.
input_vocab_size = len(train_dataset.token_vocab)
output_vocab_size = len(train_dataset.pos_vocab)
batch_size = 16
epochs = 6

# Initialize the model.
model = Tagger(input_vocab_size, output_vocab_size)
if torch.cuda.is_available():
    model = model.cuda()

# Loss function weights.
weight = torch.ones(output_vocab_size)
weight[0] = 0
if torch.cuda.is_available():
    weight = weight.cuda()
    
# Initialize loss function and optimizer.
loss_function = torch.nn.NLLLoss(weight)
optimizer = torch.optim.Adam(model.parameters())

# Main training loop.
data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                         collate_fn=collate_annotations)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_annotations)
losses = []
i = 0
for epoch in range(epochs):
    for inputs, targets, lengths in data_loader:
        optimizer.zero_grad()
        outputs, _ = model(inputs, lengths=lengths)

        outputs = outputs.view(-1, output_vocab_size)
        targets = targets.view(-1)

        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.data[0])
        if (i % 10) == 0:
            # Compute dev loss over entire dev set.
            # NOTE: This is expensive. In your work you may want to only use a 
            # subset of the dev set.
            dev_losses = []
            for inputs, targets, lengths in dev_loader:
                outputs, _ = model(inputs, lengths=lengths)
                outputs = outputs.view(-1, output_vocab_size)
                targets = targets.view(-1)
                loss = loss_function(outputs, targets)
                dev_losses.append(loss.data[0])
            avg_train_loss = np.mean(losses)
            avg_dev_loss = np.mean(dev_losses)
            losses = []
            print('Iteration %i - Train Loss: %0.6f - Dev Loss: %0.6f' % (i, avg_train_loss, avg_dev_loss), end='\r')
            torch.save(model, 'pos_tagger.pt')
        i += 1
        
torch.save(model, 'pos_tagger.final.pt')

# Collect the predictions and targets
y_true = []
y_pred = []

for inputs, targets, lengths in dev_loader:
    outputs, _ = model(inputs, lengths=lengths)
    _, preds = torch.max(outputs, dim=2)
    targets = targets.view(-1)
    preds = preds.view(-1)
    if torch.cuda.is_available():
        targets = targets.cpu()
        preds = preds.cpu()
    y_true.append(targets.data.numpy())
    y_pred.append(preds.data.numpy())

# Stack into numpy arrays
y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

# Compute accuracy
acc = np.mean(y_true[y_true != 0] == y_pred[y_true != 0])
print('Accuracy - %0.6f\n' % acc)

# Evaluate f1-score
from sklearn.metrics import f1_score
score = f1_score(y_true, y_pred, average=None)
print('F1-scores:\n')
for label, score in zip(dev_dataset.pos_vocab._id2word[1:], score[1:]):
    print('%s - %0.6f' % (label, score))

model = torch.load('pos_tagger.final.pt')

def inference(sentence):
    # Convert words to id tensor.
    ids = [[dataset.token_vocab.word2id(x)] for x in sentence]
    ids = Variable(torch.LongTensor(ids))
    if torch.cuda.is_available():
        ids = ids.cuda()
    # Get model output.
    output, _ = model(ids)
    _, preds = torch.max(output, dim=2)
    if torch.cuda.is_available():
        preds = preds.cpu()
    preds = preds.data.view(-1).numpy()
    pos_tags = [dataset.pos_vocab.id2word(x) for x in preds]
    for word, tag in zip(sentence, pos_tags):
        print('%s - %s' % (word, tag))

sentence = "sdfgkj asd;glkjsdg ;lkj  .".split()
inference(sentence)

import torch
from collections import Counter
from torch.autograd import Variable
from torch.utils.data import Dataset


class Annotation(object):
    def __init__(self):
        """A helper object for storing annotation data."""
        self.tokens = []
        self.sentiment = None


class SentimentDataset(Dataset):
    def __init__(self, fname):
        """Initializes the SentimentDataset.
        Args:
            fname: The .tsv file to load data from.
        """
        self.fname = fname
        self.annotations = self.process_tsv_file(fname)
        self.token_vocab = Vocab([x.tokens for x in self.annotations],
                                 unk_token='<unk>')

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        input = [self.token_vocab.word2id(x) for x in annotation.tokens]
        target = annotation.sentiment
        return input, target

    def process_tsv_file(self, fname):
        # Read the entire file.
        with open(fname, 'r') as f:
            lines = f.readlines()
        annotations = []
        observed_ids = set()
        for line in lines[1:]:
            annotation = Annotation()
            _, sentence_id, sentence, sentiment = line.split('\t')
            sentence_id = sentence_id
            if sentence_id in observed_ids:
                continue
            else:
                observed_ids.add(sentence_id)
            annotation.tokens = sentence.split()
            annotation.sentiment = int(sentiment)
            if len(annotation.tokens) > 0:
                annotations.append(annotation)
        return annotations


def pad(sequences, max_length, pad_value=0):
    """Pads a list of sequences.
    Args:
        sequences: A list of sequences to be padded.
        max_length: The length to pad to.
        pad_value: The value used for padding.
    Returns:
        A list of padded sequences.
    """
    out = []
    for sequence in sequences:
        padded = sequence + [0]*(max_length - len(sequence))
        out.append(padded)
    return out


def collate_annotations(batch):
    """Function used to collate data returned by CoNLLDataset."""
    # Get inputs, targets, and lengths.
    inputs, targets = zip(*batch)
    lengths = [len(x) for x in inputs]
    # Sort by length.
    sort = sorted(zip(inputs, targets, lengths),
                  key=lambda x: x[2],
                  reverse=True)
    inputs, targets, lengths = zip(*sort)
    # Pad.
    max_length = max(lengths)
    inputs = pad(inputs, max_length)
    # Transpose.
    inputs = list(map(list, zip(*inputs)))
    # Convert to PyTorch variables.
    inputs = Variable(torch.LongTensor(inputs))
    targets = Variable(torch.LongTensor(targets))
    lengths = Variable(torch.LongTensor(lengths))
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        targets = targets.cuda()
        lengths = lengths.cuda()
    return inputs, targets, lengths

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SentimentClassifier(nn.Module):
    def __init__(self,
                 input_vocab_size,
                 output_vocab_size,
                 embedding_dim=64,
                 hidden_size=64):
        """Initializes the tagger.
        
        Args:
            input_vocab_size: Size of the input vocabulary.
            output_vocab_size: Size of the output vocabulary.
            embedding_dim: Dimension of the word embeddings.
            hidden_size: Number of units in each LSTM hidden layer.
        """
        # Always do this!!!
        super(SentimentClassifier, self).__init__()

        # Store parameters
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        
        # Define layers
        self.word_embeddings = nn.Embedding(input_vocab_size, embedding_dim,
                                            padding_idx=0)
        self.rnn = nn.GRU(embedding_dim, hidden_size, dropout=0.9)
        self.fc = nn.Linear(hidden_size, output_vocab_size)
        self.activation = nn.LogSoftmax(dim=2)

    def forward(self, x, lengths=None, hidden=None):
        """Computes a forward pass of the language model.
        
        Args:
            x: A LongTensor w/ dimension [seq_len, batch_size].
            lengths: The lengths of the sequences in x.
            hidden: Hidden state to be fed into the lstm.
            
        Returns:
            net: Probability of the next word in the sequence.
            hidden: Hidden state of the lstm.
        """
        seq_len, batch_size = x.size()
        
        # If no hidden state is provided, then default to zeros.
        if hidden is None:
            hidden = Variable(torch.zeros(1, batch_size, self.hidden_size))
            if torch.cuda.is_available():
                hidden = hidden.cuda()

        net = self.word_embeddings(x)
        if lengths is not None:
            lengths_list = lengths.data.view(-1).tolist()
            net = pack_padded_sequence(net, lengths_list)
        net, hidden = self.rnn(net, hidden)
        # NOTE: we are using hidden as the input to the fully-connected layer, not net!!!
        net = self.fc(hidden)
        net = self.activation(net)

        return net, hidden

import numpy as np
from torch.utils.data import DataLoader

# Load dataset.
sentiment_dataset = SentimentDataset('train.tsv')

# Hyperparameters / constants.
input_vocab_size = len(sentiment_dataset.token_vocab)
output_vocab_size = 5
batch_size = 16
epochs = 7

# Initialize the model.
model = SentimentClassifier(input_vocab_size, output_vocab_size)
if torch.cuda.is_available():
    model = model.cuda()

# Initialize loss function and optimizer.
loss_function = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters())

# Main training loop.
data_loader = DataLoader(sentiment_dataset, batch_size=batch_size, shuffle=True,
                         collate_fn=collate_annotations)
losses = []
i = 0
for epoch in range(epochs):
    for inputs, targets, lengths in data_loader:
        optimizer.zero_grad()
        outputs, _ = model(inputs, lengths=lengths)

        outputs = outputs.view(-1, output_vocab_size)
        targets = targets.view(-1)

        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.data[0])
        if (i % 100) == 0:
            average_loss = np.mean(losses)
            losses = []
            print('Iteration %i - Loss: %0.6f' % (i, average_loss), end='\r')
        if (i % 1000) == 0:
            torch.save(model, 'sentiment_classifier.pt')
        i += 1
        
torch.save(model, 'sentiment_classifier.final.pt')

model = torch.load('sentiment_classifier.final.pt')

def inference(sentence):
    # Convert words to id tensor.
    ids = [[sentiment_dataset.token_vocab.word2id(x)] for x in sentence]
    ids = Variable(torch.LongTensor(ids))
    if torch.cuda.is_available():
        ids = ids.cuda()
    # Get model output.
    output, _ = model(ids)
    _, pred = torch.max(output, dim=2)
    if torch.cuda.is_available():
        pred = pred.cpu()
    pred = pred.data.view(-1).numpy()
    print('Sentence: %s' % ' '.join(sentence))
    print('Sentiment (0=negative, 4=positive): %i' % pred)

sentence = 'Zot zot  .'.split()
inference(sentence)

