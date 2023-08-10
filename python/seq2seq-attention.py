import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

vocab_size = 256  # ascii size
x_ = list(map(ord, "hello"))  # convert to list of ascii codes
y_ = list(map(ord, "hola"))   # convert to list of ascii codes
print("hello -> ", x_)
print("hola  -> ", y_)

x = Variable(th.LongTensor(x_))
y = Variable(th.LongTensor(y_))

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Seq2Seq, self).__init__()
        self.n_layers = 1
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.LSTM(hidden_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size)
        self.project = nn.Linear(hidden_size, vocab_size)
        self.attn = nn.Linear(2 * hidden_size, hidden_size)

    def attention(self, M:"ixbxh", h_t:"bxh", i_size, b_size)->"[bxi,bxh]":
        # bxixh * bxh -> attention [bxi]
        attention = F.softmax(th.bmm(M.t(), h_t.unsqueeze(2)))
        # bxixh * bxi -> context [bxh]
        print("M:", M.size(), "att:", attention.size())
        context = th.bmm(M.t().transpose(1, 2), attention)
        return attention.squeeze(), context.squeeze()

    def attention_func(self, encoder_outputs, decoder_hidden):
        # (batch_size*input_len*hidden_len) * (batch_size*hidden_len) -> batch_size x seq_len
        dot = torch.bmm(encoder_outputs.t(), decoder_hidden.unsqueeze(2))
        attention = F.softmax(dot.squeeze())
        # (batch_size*input_len*hidden_len)' * (batch_size*input_len) -> (batch_size*hidden_len)
        enc_attention = torch.bmm(torch.transpose(torch.transpose(encoder_outputs, 0, 1), 1, 2), attention.unsqueeze(2))
        # (batch_size*hidden_len)*(hidden_len*hidden_len) + (batch_size*hidden_len)*(hidden_len*hidden_len)
        # -> (batch_size*hidden_len)
        hidden = self.attn_tanh(self.attn_enc_linear(enc_attention.squeeze()) + self.attn_dec_linear(decoder_hidden))
        return hidden
    
    def attention(self, output_t, context):
        # Output(t) = B x H x 1
        # Context = B x T x H
        # a = B x T x 1
        a = torch.bmm(context, self.output_to_attn(output_t).unsqueeze(2))
        a = F.softmax(a.squeeze(2))  # batch_size x seq_len
        # Want to apply over context, scaled by a
        # (B x 1 x T) (B x T x H) = (B x 1 x H)
        a = a.view(a.size(0), 1, a.size(1)) 
        combined = torch.bmm(a, context).squeeze(1)
        combined = torch.cat([combined, output_t], 1)
        combined = F.tanh(self.attn(combined))
        return combined

    
    def forward(self, inputs, targets):
        # Encoder inputs and states
        initial_state = self._init_state()
        embedding = self.embedding(inputs).unsqueeze(1)
        # embedding = [seq_len, batch_size, embedding_size]
        
        # Encoder
        encoder_output, encoder_state = self.encoder(embedding, initial_state)
        # encoder_output = [seq_len, batch_size, hidden_size]
        # encoder_state  = [n_layers, seq_len, hidden_size]

        # Decoder inputs and states
        decoder_state = encoder_state
        decoder_input = Variable(th.LongTensor([[0]]))
        
        # Decoder
        outputs = []
        for i in range(targets.size()[0]):
            decoder_input = self.embedding(decoder_input)
            decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)
            
            attention, context = self.attention(encoder_output, decoder_output)
            # Project to the vocabulary size
            projection = self.project(decoder_output.view(1, -1))  # batch x vocab_size
            
            # Make prediction
            prediction = F.softmax(projection)  # batch x vocab_size
            outputs.append(prediction)
            
            # update decoder input
            _, top_i = prediction.data.topk(1)  # 1 x 1
            decoder_input = Variable(top_i)

        outputs = th.stack(outputs).squeeze()
        return outputs
    
    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return (
            Variable(weight.new(self.n_layers, batch_size, self.hidden_size).zero_()),
            Variable(weight.new(self.n_layers, batch_size, self.hidden_size).zero_())
        )

seq2seq = Seq2Seq(vocab_size, 16)
print(seq2seq)
pred = seq2seq(x, y)
print(pred)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(seq2seq.parameters(), lr=1e-3)

log = []
for i in range(1000):
    prediction = seq2seq(x, y)
    loss = criterion(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_val = loss.data[0]
    log.append(loss_val)
    if i % 100 == 0:
        print("%d loss: %s" % (i, loss_val))
        _, top1 = prediction.data.topk(1, 1)
        for c in top1.squeeze().numpy().tolist():
            print(chr(c), end=" ")
        print()

import matplotlib.pyplot as plt
plt.plot(log)
plt.ylabel('cross entropy loss')
plt.show()

