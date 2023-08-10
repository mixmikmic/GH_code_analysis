import pickle
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.autograd import Variable

# Number of words for which we are storing an embedding
vocab_size    = 220000
# Number of dimension of the embeddings
embedding_dim = 50
batch_size    = 256
input_len     = 36
epochs        = 10
print_every   = 1000
cuda          = True

def load_files():
    with open('../data/sentiment_data.pkl', 'rb') as data_file:
        data = pickle.load(data_file)

    with open('../data/sentiment_vocabulary.pkl', 'rb') as vocab_file:
        vocab = pickle.load(vocab_file)
        
    return data, vocab

def create_word_to_idx(vocab):
    items       = list(vocab.items())
    items       = sorted(items, key = lambda x: x[1], reverse = True)
    word_to_idx = {word : i + 1 for i, (word, _) in enumerate(items[:vocab_size])}
    
    return word_to_idx

def encode_data(data, word_to_idx, input_len):
    encoded_data = []
    
    # For each tweet, we compute the sequence of indices corresponding to 
    # its list of words. If the length of this sequence is smaller than 
    # input_len words, we pad it with zeros. If the sequence is longer, we 
    # cut it down to input_len words. 
    for tweet, target in data:
        encoded_tweet = [word_to_idx.get(word, 0) for word in tweet]
        len_encoding  = len(encoded_tweet) 
        if len(encoded_tweet) < input_len:
            encoded_tweet = encoded_tweet + [0] * (input_len - len_encoding)
        else:
            encoded_tweet = encoded_tweet[:input_len]
        encoded_data.append((' '.join(tweet), encoded_tweet, target))
        
    return encoded_data

def load_data(vocab_size, input_len, test_proportion = 0.2):
    data, vocab   = load_files()
    word_to_idx   = create_word_to_idx(vocab)
    encoded_data  = encode_data(data, word_to_idx, input_len)
    # We split the data into a training set and a test set.
    training_size = int(len(encoded_data) * (1 - test_proportion))  
    random.shuffle(encoded_data)
    training_data = encoded_data[:training_size]
    test_data     = encoded_data[training_size:]
    
    return training_data, test_data

def batch_to_tensor(batch):
    tweets  = [tweet for tweet, _, _ in batch]
    inputs  = torch.LongTensor([input for _, input, _ in batch])
    targets = torch.LongTensor([target for _, _, target in batch])
    
    return tweets, inputs, targets

def batch_generator(data, batch_size, shuffle = True):
    if shuffle:
        data = random.sample(data, len(data))
        
    return (batch_to_tensor(data[i: i + batch_size]) for i in range(0, len(data), batch_size))

def evaluate_model(cnn, criterion, train_data, test_data, batch_size):
    def evaluate_model_data(data):
        batch_number     = 0
        total_loss       = 0
        total_correct    = 0
        total_prediction = 0
        for _, inputs, targets in batch_generator(data, batch_size, shuffle = False):
            inputs            = Variable(inputs)
            targets           = Variable(targets)
            inputs            = inputs.cuda() if cuda else inputs
            targets           = targets.cuda() if cuda else targets
            predictions       = cnn(inputs)
            loss              = criterion(predictions, targets)
            total_loss       += loss.cpu().data[0]
            batch_number     += 1
            pred_classes      = predictions.max(dim = 1)[1]
            total_prediction += predictions.size()[0]
            total_correct    += (pred_classes == targets).cpu().sum().data[0]
        average_loss     = total_loss / batch_number
        average_accuracy = total_correct / total_prediction
        
        return average_loss, average_accuracy
    
    return evaluate_model_data(train_data), evaluate_model_data(test_data)

def print_model_evaluation(cnn, epoch, criterion, train_data, test_data, batch_size):
    cnn.eval()
    evaluation = evaluate_model(cnn, criterion, train_data, test_data, batch_size)
    cnn.train()
    print(
        f'[{epoch + 1:3}] ' 
        f'train loss: {evaluation[0][0]:.4f}, train accuracy: {100 * evaluation[0][1]:.3f}%, '
        f'test loss: {evaluation[1][0]:.4f}, test accuracy: {100 * evaluation[1][1]:.3f}%'
    )

class CNN(nn.Module):
    def __init__(self, vocab_size, input_len, embedding_dim):
        super(CNN, self).__init__()
        self.input_len     = input_len
        self.embedding_dim = embedding_dim
        self.embedding     = nn.Embedding(vocab_size + 1, embedding_dim)
        self.conv1         = nn.Conv1d(embedding_dim, 64, 3, padding = 1)
        self.bn1           = nn.BatchNorm1d(64)
        self.dropout1      = nn.Dropout(p = 0.8)
        self.conv2         = nn.Conv1d(64 , 64 , 3, padding = 1)
        self.bn2           = nn.BatchNorm1d(64)
        self.dropout2      = nn.Dropout(p = 0.8)
        self.conv3         = nn.Conv1d(64 , 128, 3, padding = 1)
        self.bn3           = nn.BatchNorm1d(128)
        self.dropout3      = nn.Dropout(p = 0.8)
        self.conv4         = nn.Conv1d(128, 128, 3, padding = 1)
        self.bn4           = nn.BatchNorm1d(128)
        self.dropout4      = nn.Dropout(p = 0.8)
        self.linear1       = nn.Linear(128 * 9, 256)
        self.bn5           = nn.BatchNorm1d(256)
        self.dropout5      = nn.Dropout(p = 0.8)
        self.linear2       = nn.Linear(256, 256)
        self.bn6           = nn.BatchNorm1d(256)
        self.dropout6      = nn.Dropout(p = 0.8)
        self.linear3       = nn.Linear(256, 2)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2).contiguous()
        x = self.dropout1(self.bn1(F.relu(self.conv1(x))))
        x = self.dropout2(self.bn2(F.relu(self.conv2(x))))
        x = F.avg_pool1d(x, 2)
        x = self.dropout3(self.bn3(F.relu(self.conv3(x))))
        x = self.dropout4(self.bn4(F.relu(self.conv4(x))))
        x = F.avg_pool1d(x, 2)
        x = x.view(-1, 9 * 128)
        x = self.dropout5(self.bn5(F.relu(self.linear1(x))))
        x = self.dropout6(self.bn6(F.relu(self.linear2(x))))
        x = F.log_softmax(self.linear3(x), dim = 1)
        
        return x

train_data, test_data = load_data(vocab_size, input_len)
cnn                   = CNN(vocab_size, input_len, embedding_dim)
cnn                   = cnn.cuda() if cuda else cnn
criterion             = nn.NLLLoss()
optimizer             = optim.Adam(cnn.parameters())

print_model_evaluation(cnn, 0, criterion, train_data, test_data, batch_size)
for epoch in range(epochs):
    total_loss   = 0
    running_loss = 0
    for i, (_, inputs, targets) in enumerate(batch_generator(train_data, batch_size)):
        optimizer.zero_grad()
        inputs        = Variable(inputs)
        targets       = Variable(targets)
        inputs        = inputs.cuda() if cuda else inputs
        targets       = targets.cuda() if cuda else targets
        predictions   = cnn(inputs)
        loss          = criterion(predictions, targets)
        loss_value    = loss.cpu().data[0]
        running_loss += loss_value
        loss.backward()
        optimizer.step()
        
        if i % print_every == print_every - 1:
            print(f'\t[{i + 1:6}] running_loss: {running_loss / print_every:.4f}')
            running_loss = 0

    print_model_evaluation(cnn, epoch + 1, criterion, train_data, test_data, batch_size)

