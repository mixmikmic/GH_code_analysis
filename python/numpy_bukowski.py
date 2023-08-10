## Import libaries.
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import operator
import nltk
import sys
import os

get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0)
sns.set_style("whitegrid")

size = 8000
unknown_token = 'UNKNOWN_TOKEN'
start_symbol = 'START_SYMBOL'
stop_symbol = 'STOP_SYMBOL'

## Collect input files.
files = []
for file in os.listdir("."):
    if file.endswith(".txt"):
        files.append(file)

## Create empty data frame.
content = pd.DataFrame({'line' : [], 'text' : [], 'tokens' : [], 'poem' : []})

## Fill dataframe from files.
for file in files:
    print "Processing file", file
    for l in open(file, 'rb'):
        line = l.strip('\n')
        if l[0].isdigit():
            data = line.split('   ')
            if len(data) > 1:
                #sentence = start_symbol + ' ' + data[1] + ' ' + stop_symbol
                sentence = data[1]
                tokens = nltk.word_tokenize(sentence.decode('utf-8').lower())
                ## Add start and stop symbols.
                tokens.insert(0, start_symbol)
                tokens.append(stop_symbol)
                content = content.append({'line' : int(data[0]), 'text' : data[1], 'tokens' : tokens}, ignore_index = True)

## Make the 'line' content numeric 
content['line'] = pd.to_numeric(content['line'])
print "Processed a total of %d lines." %content.shape[0]

count = 1
poem = 1
for i, row in content.iterrows():
    if row['line'] >= count:
        count = row['line']
        content.loc[i, 'poem'] = poem
    else:
        count = 1
        poem += 1
        content.loc[i, 'poem'] = poem
print "Parsed %d distinct poems." %poem

## Store results on pickle.
content.to_pickle('data/content.pkl')

## Get word frequency.
sentences = [content.loc[i, 'tokens'] for i in content.index]
flattened = itertools.chain.from_iterable(sentences) 
word_freq = nltk.FreqDist(flattened)
print "Found %d unique words tokens." % len(word_freq.items())

## Limit vocabulary to most common words.
vocab = word_freq.most_common(size-1)
word_list = [x[0] for x in vocab]
word_list.append(unknown_token)
word_list
word_index = dict([(w,i) for i,w in enumerate(word_list)])

print "Using vocabulary size %d." % size
print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

## Replace words not in dictionary with the unknown token.
for i, row in content.iterrows():
    for j, tkn in enumerate(row['tokens']):
        if tkn not in word_list:
            content.loc[i, 'tokens'][j] = unknown_token
            
print 'Sample processed poem:'
for index, row in content[content['poem'] == 10].iterrows():
    print ' '.join(row['tokens'])

## Convert the info into training data.
X_train = np.asarray([[word_index[w] for w in sent[:-1]] for sent in content['tokens'].tolist()])
y_train = np.asarray([[word_index[w] for w in sent[1:]] for sent in content['tokens'].tolist()])

## Show an example.
x_example, y_example = X_train[39], y_train[39]
print 'Example Input data: ', 
input_sent = [word_list[i] for i in x_example]
print ' '.join(x for x in input_sent)

print 'Example Output data:', 
output_sent = [word_list[j] for j in y_example]
print ' '.join(x for x in output_sent)

## Define our softmax function.
def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

class RNNNumpy:
    def __init__(self, w_dim, h_dim=100, bptt_max=4):
        self.w_dim = w_dim
        self.h_dim = h_dim
        self.bptt_max = bptt_max
        ## Randomly initialize network parameters.
        ## Set to uniform between [-1/n, 1/n], where
        ## 'n' is the size of incoming connections.
        self.U = np.random.uniform(-np.sqrt(1./w_dim), np.sqrt(1./w_dim), (h_dim, w_dim))
        self.V = np.random.uniform(-np.sqrt(1./h_dim), np.sqrt(1./h_dim), (w_dim, h_dim))
        self.W = np.random.uniform(-np.sqrt(1./h_dim), np.sqrt(1./h_dim), (h_dim, h_dim))
        
    def forward_propagation(self, x):
        ## Length of inputs.
        T = len(x)
        ## Generate matrix for all hidden states and all outputs.
        s = np.zeros((T+1, self.h_dim))
        o = np.zeros((T, self.w_dim))
        ## Iterate though the steps.
        for t in np.arange(T):
            # s[t] = Ux[t] = Ws[t-1]
            s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
            # o[t] = softmax(Vs[t])
            o[t] = softmax(self.V.dot(s[t]))
        return [o, s]
    
    def predict(self, x):
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)
    
    def total_loss_function(self, x, y):
        L = 0
        ## Calculate loss on each sentence.
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            ## Select predictions for correct words.
            correct_preds = o[np.arange(len(y[i])), y[i]]
            L -= np.sum(np.log(correct_preds))
        return L
    
    def loss_function(self, x, y):
        N = np.sum(len(y_i) for y_i in y)
        return self.total_loss_function(x,y)/N
    
    def bptt(self, x, y):
        T = len(y)
        ## Start with forward-propagation.
        o, s = self.forward_propagation(x)
        dLdU = np.zeros_like(self.U)
        dLdV = np.zeros_like(self.V)
        dLdW = np.zeros_like(self.W)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1
        ## Go over observations, from end to start.
        for t in np.arange(T)[::-1]:
            dLdV = np.outer(delta_o[t], s[t].T)
            ## Initial delta.
            delta_t = np.inner(self.V.T, delta_o[t]) * (1 - s[t]**2)
            ## Now we do the back-propagation.
            for step in np.arange(max(0, t - self.bptt_max), t+1)[::-1]:
                dLdW = np.outer(delta_t, s[step - 1]) 
                dLdU[:, x[step]] += delta_t
                ## Update delta for next iteration.
                delta_t = np.inner(self.W.T, delta_t) * (1 - s[step - 1]**2)
        return [dLdU, dLdV, dLdW]
    
    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        ## Calculate the parameters with bptt.
        bptt_gradients = model.bptt(x,y)
        ## Parameters to check.
        model_params = ['U','V','W']
        ## Perform 'manual' check on each parameter.
        for pid, pname in enumerate(model_params):
            parameter = operator.attrgetter(pname)(self)
            print 'Performing gradient check on %s with size %d.' %(pname, np.prod(parameter.shape))
            ## Iterate over elements of the parameter matrix.
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                ## Store original value.
                original_val = parameter[ix]
                ## Estimate gradient manually.
                parameter[ix] = original_val + h
                grad_plus = model.total_loss_function([x], [y])
                parameter[ix] = original_val - h
                grad_minus = model.total_loss_function([x], [y])
                estimated_gradient = (grad_plus - grad_minus) / (2*h)
                ## Reset parameter to original value.
                parameter[ix] = original_val
                ## Estimate gradient with bptt.
                backprop_gradient = bptt_gradients[pid][ix]
                ## Calculate relative error.
                relative_error = np.abs(backprop_gradient - estimated_gradient) /                 (np.abs(backprop_gradient) + np.abs(estimated_gradient))
                ## If the error is to large, do not pass the test.
                if relative_error > error_threshold:
                    print 'Gradient check ERROR: parameter=%s, index=%s.' %(pname, ix)
                    print 'Relative Error: %f.' %relative_error
                it.iternext()
            print 'Gradient check for parameter %s passed! :)' %pname
            
    def sgd_step(self, x, y, learning_rate):
        ## Calculate gradients with bptt.
        dLdU, dLdV, dLdW = self.bptt(x,y)
        ## Update according to gradient and learning rate.
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW

def train_with_sgd(model, X_train, y_train, learning_rate=0.005, epochs=100, evaluate_loss_after=5):
    ## List to keep track of losses.
    losses = []
    num_examples_seen = 0
    
    for epoch in range(epochs):
        if(epoch % evaluate_loss_after == 0):
            loss = model.loss_function(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%H:%M:%S')
            print '%s: Loss after num_examples=%d & epoch=%d: %f.' %(time, num_examples_seen, epoch, loss)
            
            ## Adjust learning rate if loss increases.
            if(len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print 'Setting learning rate to %f.' %learning_rate
            
            ## For each training example...
        for i in range(len(y_train)):
            ## do one SGD step.
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1
    
    return losses

## Running time of one SGD.
model = RNNNumpy(size)
get_ipython().magic('timeit model.sgd_step(X_train[10], y_train[10], 0.005)')

print "Expected random prediction loss: %f." % np.log(size)

## Verify that indeed the BPTT algorithm reduces the error over time.
losses = train_with_sgd(model, X_train, y_train, epochs=10, evaluate_loss_after=1)

plt.plot([l[1] for l in losses], 'go')
plt.plot([l[1] for l in losses], 'g-')
plt.axhline(np.log(size))
plt.title('Random Prediction vs. Model Errors')

def generate_sentence(model):
    # We start the sentence with the start token
    new_sentence = [word_index[start_symbol]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_index[stop_symbol]:
        next_word_probs = model.forward_propagation(new_sentence)
        sampled_word = word_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == word_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1][0])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [word_list[x] for x in new_sentence[1:-1]]
    return sentence_str

num_sentences = 10
senten_min_length = 3

for i in range(num_sentences):
    sent = []
    # We want long sentences, not sentences with one or two words
    while len(sent) < senten_min_length:
        sent = generate_sentence(model)
    print " ".join(sent)

