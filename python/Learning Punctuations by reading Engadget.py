import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import forward_tracer, backward_tracer, Char2Vec, num_flat_features

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import numpy as np

from tqdm import tqdm

from IPython.display import clear_output

import os

from bs4 import BeautifulSoup

def get_content(fn):
    with open(fn, 'r') as f:
        source = ""
        for line in f:
            source += line
    return source

def source_gen(path="../engadget_data/"):
    for child, folders, files in os.walk(path):
        for fn in files:
            if fn[0] is ".":
                pass
            else: 
                src = get_content(path + fn)
                soup = BeautifulSoup(src, 'html.parser')
                src = soup.getText()
                yield fn, src

for fn, text in source_gen():
    print(text)
    break

import math

def batch_gen(seq_length, source):
    s_l = len(source)
    b_n = math.ceil(s_l/seq_length)
    s_pad = source + " " * (b_n * seq_length - s_l)
    for i in range(b_n):
        yield s_pad[i*seq_length: (i+1)*seq_length]

def get_chars():
    step = 0
    freq = {}
    keys = []
    for file_name, source in tqdm(source_gen()):

        for char in source:
            try:
                freq[char] += 1
            except KeyError:
                freq[char] = 1
                keys.append(char)
        #if step%10000 == 9999:
        #    print(str(step) + ": ln: " + str(len(keys)) + str(["".join(keys)]))
    
    return keys, freq
ks, freqs = get_chars()
print("".join(ks))

order = np.argsort([freqs[k] for k in ks])[::-1]
chars_ordered = "".join(np.array([k for k in ks])[order])
print(chars_ordered[:140])

plt.title('Frequency of each character')
plt.plot(np.array([math.log10(freqs[k]) for k in ks])[order])
plt.ylabel('$\log_10$ frequency')
plt.xlabel('character index')
plt.show()

input_chars = list(" ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz01234567890")
output_chars = ["<nop>", "<cap>"] + list(".,;:?!\"'")

class GruRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers=1, bi=False):
        super(GruRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = layers
        self.bi_mul = 2 if bi else 1
        
        self.encoder = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(input_size, hidden_size, self.layers, bidirectional=bi)
        self.decoder = nn.Linear(hidden_size * self.bi_mul, output_size)
        self.softmax = F.softmax
        
    def forward(self, x, hidden):
        #embeded = self.encoder(x)
        embeded = x
        #print(embeded.view(-1, 1, self.input_size).size())
        #print(hidden.size())
        gru_output, hidden = self.gru(embeded.view(-1, 1, self.input_size), hidden.view(self.layers * self.bi_mul, -1, self.hidden_size))
        #print(gru_output.size())
        output = self.decoder(gru_output.view(-1, self.hidden_size * self.bi_mul))
        return output, hidden
    
    def init_hidden(self, random=False):
        if random:
            return Variable(torch.randn(self.layers * self.bi_mul, self.hidden_size))
        else:
            return Variable(torch.zeros(self.layers * self.bi_mul, self.hidden_size)) 
"""
input_size = 105
hidden_size = 105
output_size = 105
layers = 2

gRNN = GruRNN(input_size, hidden_size, output_size, layers)

gRNN(Variable(torch.FloatTensor(10000, 105)),
     Variable(torch.FloatTensor(layers, 105)))"""

class Engadget():
    def __init__(self, model, char2vec=None, output_char2vec=None):
        self.model = model
        if char2vec is None:
            self.char2vec = Char2Vec()
        else:
            self.char2vec = char2vec
            
        if output_char2vec is None:
            self.output_char2vec = self.char2vec
        else:
            self.output_char2vec = output_char2vec
            
        self.loss = 0
        self.losses = []
    
    def init_hidden_(self, random=False):
        self.hidden = model.init_hidden(random)
        return self
    
    def save(self, fn="GRU_Engadget.tar"):
        torch.save({
            "hidden": self.hidden, 
            "state_dict": model.state_dict(),
            "losses": self.losses
                   }, fn)
    
    def load(self, fn):
        checkpoint = torch.load(fn)
        self.hidden = checkpoint['hidden']
        model.load_state_dict(checkpoint['state_dict'])
        self.losses = checkpoint['losses']
    
    def setup_training(self, learning_rate):
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        self.init_hidden_()
        
    def reset_loss(self):
        self.loss = 0
        
    def forward(self, input_text, target_text):
        
        self.hidden = self.hidden.detach()
        
        self.optimizer.zero_grad()
        self.next_(input_text)
        target_vec = Variable(self.output_char2vec.char_code(target_text))
        self.loss += self.loss_fn(self.output, target_vec)
        
    def descent(self):
        if self.loss is 0:
            print(self.loss)
            print('Warning: loss is zero.')
            return
        
        self.loss.backward()
        self.optimizer.step()
        self.losses.append(self.loss.cpu().data.numpy())
        self.reset_loss()
    
    def embed(self, input_data):
        self.embeded = Variable(self.char2vec.one_hot(input_data))
        return self.embeded
        
    def next_(self, input_text):
        self.output, self.hidden = self.model(self.embed(input_text), self.hidden)
        return self
    
    def softmax_(self, temperature=0.5):
        self.softmax = self.model.softmax(self.output/temperature)
        return self
    
    def output_chars(self, start=None, end=None):
        indeces = torch.multinomial(self.softmax[start:end]).view(-1)
        return self.output_char2vec.vec2list(indeces)

def apply_punc(text_input, text_output):
    result = ""
    for char1, char2 in zip(text_input, text_output):
        if char2 == "<cap>":
            result += char1.upper()
        elif char2 != "<nop>":
            result += char1 + char2
        else:
            result += char1
    return result


result = apply_punc("t s", ['<cap>', '<nop>', ','])
assert(result == "T s,")

def extract_punc(string_input, input_chars, output_chars):
    input_source = []
    output_source = []
    for i, char in enumerate(string_input):
        # print(i, char)
        if char.isupper() and len(output_source) > 0:
            output_source.append("<cap>")
            input_source.append(char.lower())
        elif char in output_chars and len(output_source) > 0:
            output_source[-1] = char
        elif char in input_chars:
            input_source.append(char)
            output_source.append("<nop>")
    return input_source, output_source

i, o = extract_punc("This's a simple ATI chassis.", input_chars, output_chars)
result = apply_punc("".join(i), o)
print(result)

char2vec = Char2Vec(chars=input_chars, add_unknown=True)
output_char2vec = Char2Vec(chars = output_chars)
input_size = char2vec.size 
output_size = output_char2vec.size

print("input_size is: " + str(input_size) + "; ouput_size is: " + str(output_size))
hidden_size = input_size
layers = 1

model = GruRNN(input_size, hidden_size, output_size, layers=layers, bi=True)
egdt = Engadget(model, char2vec, output_char2vec)
egdt.load('./Gru_Engadget_1_layer_bi.tar')

learning_rate = 2e-3
egdt.setup_training(learning_rate)

model.zero_grad()
egdt.reset_loss()

seq_length = 200

for epoch_num in range(40):
    
    step = 0
    for file_name, source in tqdm(source_gen()):
        
        for source_ in batch_gen(seq_length, source):
            
            step += 1
            
            input_source, output_source = extract_punc(source_, egdt.char2vec.chars, egdt.output_char2vec.chars)
            
            try:
                egdt.forward(input_source, output_source)
                if step%1 == 0:
                    egdt.descent()
                    
            except KeyError:
                print(source)
                raise KeyError
            

            if step%400 == 399:
                clear_output(wait=True)
                print('Epoch {:d}'.format(epoch_num))

                egdt.softmax_()

                fig = plt.figure(figsize=(16, 8))
                fig.subplots_adjust(hspace=0.0625)
                plt.subplot(131)
                plt.title("Input")
                plt.imshow(egdt.embeded[:130].data.byte().numpy(), cmap="Greys_r", interpolation="none")
                plt.subplot(132)
                plt.title("Output")
                im = plt.imshow(egdt.output[:20].data.byte().numpy(), cmap="Greys_r", interpolation="none")
                cb = plt.colorbar(im, fraction=0.08); cb.outline.set_linewidth(0)
                plt.subplot(133)
                plt.title("Softmax Output")
                im = plt.imshow(egdt.softmax[:20].cpu().data.numpy(), interpolation="none")
                cb = plt.colorbar(im, fraction=0.08); cb.outline.set_linewidth(0)
                plt.show()

                plt.figure(figsize=(10, 3))
                plt.title('Training loss')
                plt.plot(egdt.losses, label="loss", linewidth=3, alpha=0.4)
                plt.show()
                
                # print(source_)
                
                result = apply_punc(input_source, egdt.output_chars())
                print(result)

egdt.save('./data/Gru_Engadget_1_layer_bi.tar')

from ipywidgets import widgets
from IPython.display import display

def predict_next(input_text, gen_length=None, temperature=0.05):
    
    if gen_length is None: 
        gen_length = len(input_text)
    
    clear_output(wait=True)
    #egdt = Engadget(model).init_hidden_(random=True)
    
    egdt.init_hidden_()
    egdt.next_(input_text)
    egdt.softmax_()
    output = egdt.output_chars()
    
    #print(output)
    result = apply_punc(input_text, output)
    print(result)
    
    plt.figure(figsize=(12, 9))
    plt.subplot(311)
    plt.title("Input")
    plt.imshow(egdt.embeded[:130].data.byte().numpy().T, cmap="Greys_r", interpolation="none")
    plt.subplot(312)
    plt.title("Output")
    plt.imshow(egdt.output[:130].data.byte().numpy().T, interpolation="none")
    plt.subplot(313)
    plt.title("Softmax")
    plt.imshow(egdt.softmax[:130].cpu().data.numpy().T, interpolation="none")
    plt.show()

predict_next("   this wont be a simple sentense it doesnt have puntuation yet the network can add", 200, 1)

text_input = widgets.Text()
display(text_input)

def handle_submit(sender):
    #print(text_input.value)
    predict_next(text_input.value, 2000, temperature=0.5)
    
text_input.on_submit(handle_submit)







