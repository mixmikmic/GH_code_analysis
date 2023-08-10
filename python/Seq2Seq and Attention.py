# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('text8.zip', 31344016)

def read_data(filename):
  with zipfile.ZipFile(filename) as f:
    name = f.namelist()[0]
    data = tf.compat.as_str(f.read(name))
  return data
  
text = read_data(filename)
print('Data size %d' % len(text))

valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])

vocabulary_size = len(string.ascii_lowercase) + 2 # [a-z] + ' ' + #(end of sentence)
first_letter = ord(string.ascii_lowercase[0])

def char2id(char):
  if char in string.ascii_lowercase:
    return ord(char) - first_letter + 2
  elif char == ' ':
    return 1
  elif char=='#':
    return 0
  else:
    print('Unexpected character: %s' % char)
    return 0
  
def id2char(dictid):
    if dictid > 1:
        return chr(dictid + first_letter - 2)
    elif dictid==1:
        return ' '
    elif dictid==0:
        return '#'



print(char2id('a'), char2id('z'), char2id(' '), char2id('ï'))
print(id2char(1), id2char(26), id2char(0))

def reverse(alist):
    newlist = []
    for i in range(1, len(alist) + 1):
        newlist.append(alist[-i])
    return newlist

batch_size=64
num_unrollings=14

class BatchGenerator(object):
  def __init__(self, text, batch_size, num_unrollings):
    self._text = text
    self._text_size = len(text)
    self._batch_size = batch_size
    self._num_unrollings = num_unrollings
    segment = self._text_size // batch_size
    self._cursor = [ offset * segment for offset in range(batch_size)]
    self._last_batch = self._next_batch()
  
  def _next_batch(self):
    """Generate a single batch from the current cursor position in the data."""
    batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
    for b in range(self._batch_size):
      batch[b, char2id(self._text[self._cursor[b]])] = 1.0
      self._cursor[b] = (self._cursor[b] + 1) % self._text_size
    return batch
  
  def next(self):
    """Generate the next array of batches from the data. The array consists of
    the last batch of the previous array, followed by num_unrollings new ones.
    """
    batches = [self._last_batch]
    batches=[]
    for step in range(self._num_unrollings-1):
      batches.append(self._next_batch())
    #The EOS character for each batch
    
    #Generating the output batches by reversing each word in a num_unrolling size sentence
    output_batches=[]
    for step in range(self._num_unrollings-1):
        output_batches.append(np.zeros(shape=(self._batch_size, vocabulary_size),dtype=np.float))
    for b in range(self._batch_size):
        words=[]
        #Will store each of characters for words, is emptied when a space is encountered
        array=[]
        for i in range(self._num_unrollings-1):
            if(np.argmax(batches[i][b,:])!=1):
                array.append(np.argmax(batches[i][b,:]))
            else:
                array=reverse(array)
                words.extend(array)
                words.append(1)
                array=[]
        array=reverse(array)
        words.extend(array)
        for i in range(self._num_unrollings-1):
            output_batches[i][b,words[i]]=1
        
    last_batch=np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
    #Set the last batch character to EOS for the last batch
    
    last_batch[:,0]=1
    batches.append(last_batch)
    output_batches.append(last_batch)
    self._last_batch = batches[-1]
    return batches,output_batches

def characters(probabilities):
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (most likely) character representation."""
  return [id2char(c) for c in np.argmax(probabilities, 1)]

def batches2string(batches):
  """Convert a sequence of batches back into their (most likely) string
  representation."""
  s = [''] * batches[0].shape[0]
  for b in batches:
    s = [''.join(x) for x in zip(s, characters(b))]
  return s

train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, num_unrollings)
batches,output_batches=train_batches.next()
print(batches2string(batches))
print(batches2string(output_batches))

#Split every 20 

def logprob(predictions, labels):
  """Log-probability of the true labels in a predicted batch."""
  predictions[predictions < 1e-10] = 1e-10
  return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution):
  """Sample one element from a distribution assumed to be an array of normalized
  probabilities.
  """
  r = random.uniform(0, 1)
  s = 0
  for i in range(len(distribution)):
    s += distribution[i]
    if s >= r:
      return i
  return len(distribution) - 1

def sample(prediction):
  """Turn a (column) prediction into 1-hot encoded samples."""
  p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
  p[0, sample_distribution(prediction[0])] = 1.0
  return p

def random_distribution():
  """Generate a random column of probabilities."""
  b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
  return b/np.sum(b, 1)[:,None]


def accuracy(labels,predictions):
    return np.sum(np.argmax(labels,axis=1)==np.argmax(predictions,axis=1))/labels.shape[0]

num_nodes =256
#dropout=1.0

class lstm:
    def __init__(self,input_size):
        self.xx = tf.Variable(tf.truncated_normal([input_size, num_nodes * 4], -0.1, 0.1))
        self.mm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes * 4], -0.1, 0.1))
        self.bb = tf.Variable(tf.zeros([1, num_nodes * 4]))


                          
    def lstm_cell(self,i,o,state):
        global dropout
        #i=tf.nn.dropout(i,keep_prob=dropout)
        matmuls = tf.matmul(i, self.xx)+ tf.matmul(o, self.mm) + self.bb        
        input_gate  = tf.sigmoid(matmuls[:, 0 * num_nodes : 1 * num_nodes])
        forget_gate = tf.sigmoid(matmuls[:, 1 * num_nodes : 2 * num_nodes])
        update      =            matmuls[:, 2 * num_nodes : 3 * num_nodes]
        output_gate = tf.sigmoid(matmuls[:, 3 * num_nodes : 4 * num_nodes])
        state       = forget_gate * state + input_gate * tf.tanh(update)
        output=output_gate * tf.tanh(state)
        return output, state



graph = tf.Graph()
with graph.as_default():

    #LSTM for encoder and decoder
    encoder_lstm=lstm(vocabulary_size)
    decoder_lstm=lstm(vocabulary_size)
    #State saving across unrollings
    saved_state=tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    saved_output=tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False) 
    state=saved_state
    output=saved_output
    
    
    reset_state = tf.group(
        output.assign(tf.zeros([batch_size, num_nodes])),
        state.assign(tf.zeros([batch_size, num_nodes])),
        )
    
    # Classifier weights and biases.
    w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
    b = tf.Variable(tf.zeros([vocabulary_size]))
    
    #Define our train input, decoder output variables
    train_inputs=[]
    decoder_inputs=[]
    outputs=[]
    
    for i in range(num_unrollings):
        train_inputs.append(tf.placeholder(tf.float32,shape=[batch_size,vocabulary_size]))
        decoder_inputs.append(tf.placeholder(tf.float32,shape=[batch_size,vocabulary_size]))
        
        
    #Function Definition
    #The Encoder function
    def encoder(train_input,output,state):
        '''
        Args:       
        
        train_input : array of size num_unrolling, each array element is a Tensor of dimension batch_size,
        vocabulary size.
        
        Returns:
        
        output : Output of LSTM aka Hidden State
        state : Cell state of the LSTM
        
        '''
        i = len(train_inputs) - 1
        while i >= 0:
            output, state = encoder_lstm.lstm_cell(train_input[i],output,state)
            i=i-1
        #Return the last output of the lstm cell for decoding
        return output ,state
    
  

    def training_decoder(decoder_input,output,state):
        outputs=[]
        #Predict the first character using the EOS Tag. We use EOS tag as the start tag
        output, state = decoder_lstm.lstm_cell(decoder_input[-1],output,state)
        outputs.append(output)
        #Now predict the next outputs using the training labels itself. Using y(n-1) to predict y(n)
        for i in decoder_input[0:-1]:
            output,state=decoder_lstm.lstm_cell(i,output,state)
            outputs.append(output)
        return outputs,output,state
    
    
    def inference_decoder(go_char,decode_steps,output,state):
        outputs=[]
        #First input to decoder is the the Go Character
        output,state=decoder_lstm.lstm_cell(go_char,output,state)
        outputs.append(output)
        for i in range(decode_steps-1):
            #Feed the previous output as the next decoder input
            decoder_input=tf.nn.softmax(tf.nn.xw_plus_b(output, w, b))
            output,state=decoder_lstm.lstm_cell(decoder_input,output,state)
            outputs.append(output)
        return outputs,output,state
    

        
    
    #Model Definition

    output,state=encoder(train_inputs,output,state)

    outputs,output,state=training_decoder(decoder_inputs,output,state)
    


    with tf.control_dependencies([saved_state.assign(state),
                                saved_output.assign(output),
                                    ]):
        logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), w, b)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                    labels=tf.concat(decoder_inputs, 0), logits=logits))
        
    #Loss function and optimizer
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
                            10.0, global_step, 5000, 0.1, staircase=True)
    optimizer = tf.train.AdamOptimizer()
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(
                    zip(gradients, v), global_step=global_step)
    
    # Predictions.
    train_prediction = tf.nn.softmax(logits)
    
    
    
    #Sample Prediction
    sample_input=[]
    sample_outputs=[]

    for i in range(num_unrollings):
        sample_input.append(tf.placeholder(tf.float32,shape=[1,vocabulary_size]))
        
    sample_saved_state=tf.Variable(tf.zeros([1, num_nodes]), trainable=False)
    sample_saved_output=tf.Variable(tf.zeros([1, num_nodes]), trainable=False)
    

    
    sample_output=sample_saved_output
    sample_state=sample_saved_state

    
    
    reset_sample_state = tf.group(
        sample_output.assign(tf.zeros([1, num_nodes])),
        sample_state.assign(tf.zeros([1, num_nodes])),

        )
    

    sample_output,sample_state=encoder(sample_input,sample_output,sample_state)
    sample_decoder_outputs,sample_output,sample_state=inference_decoder(sample_input[-1],num_unrollings,sample_output,sample_state)

    with tf.control_dependencies([sample_saved_output.assign(sample_output),
                                sample_saved_state.assign(sample_state),
                               ]):
        for d in sample_decoder_outputs:
                sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(d, w, b))
                sample_outputs.append(sample_prediction)
    
        

num_steps = 20000
summary_frequency = 1000

with tf.Session(graph=graph) as session:
  global dropout
  tf.global_variables_initializer().run()
  print('Initialized')
  mean_loss = 0
  for step in range(num_steps):
    batches,output_batches = train_batches.next()
    feed_dict = dict()
    dropout=0.5
    
    for i in range(num_unrollings):
        #Feeding input from reverse according to https://arxiv.org/abs/1409.3215
        feed_dict[train_inputs[i]]=batches[i]
        feed_dict[decoder_inputs[i]]=output_batches[i]

        
    _, l, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    #reset_state.run()

    if step % (summary_frequency ) == 0:
        dropout=1
        print('-'*80)
        print('Step '+str(step))
        print('Loss '+str(l))
        
        labels=np.concatenate(list(output_batches)[:])
#        print(characters(labels))
#       print(characters(predictions))
        print('Batch Accuracy: %.2f' % float(accuracy(labels,predictions)*100))
        num_validation = valid_size // num_unrollings
        reset_sample_state.run()
        sum_acc=0
        for _ in range(num_validation):
            valid,valid_output=valid_batches.next()
            valid_feed_dict=dict()
            for i in range(num_unrollings):
                valid_feed_dict[sample_input[i]]=valid[i]
            sample_pred=session.run(sample_outputs,feed_dict=valid_feed_dict)
            labels=np.concatenate(list(valid_output)[:],axis=0)
            pred=np.concatenate(list(sample_pred)[:],axis=0)
            sum_acc=sum_acc+accuracy(labels,pred)
        val_acc=sum_acc/num_validation
        print('Validation Accuracy: %0.2f'%(val_acc*100))
        print('Input Test String '+str(batches2string(valid)))
        print('Output Prediction'+str(batches2string(sample_pred)))
        print('Actual'+str(batches2string(valid_output)))
        

graph = tf.Graph()
with graph.as_default():
    #Variables and placeholders
    #
    #Weights for the hidden states accross time
    attn_weights=tf.Variable(tf.truncated_normal([num_nodes], -0.1, 0.1))
    #Weights for the context(hidden state) at time t-1
    prev_hidden_weights=tf.Variable(tf.truncated_normal([num_nodes], -0.1, 0.1))
    #LSTM for encoder and decoder
    encoder_lstm=lstm(vocabulary_size)
    #feed decoder Y(t-1) and attention context
    decoder_lstm=lstm(num_nodes+vocabulary_size)

    #State saving across unrollings
    saved_state=tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    saved_output=tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False) 
    state=saved_state
    output=saved_output
    
    
    reset_state = tf.group(
        output.assign(tf.zeros([batch_size, num_nodes])),
        state.assign(tf.zeros([batch_size, num_nodes])),
        )
    
    # Classifier weights and biases.
    w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
    b = tf.Variable(tf.zeros([vocabulary_size]))
    
    #Define our train input, decoder output variables
    train_inputs=[]
    decoder_inputs=[]
    outputs=[]
    
    for i in range(num_unrollings):
        train_inputs.append(tf.placeholder(tf.float32,shape=[batch_size,vocabulary_size]))
        decoder_inputs.append(tf.placeholder(tf.float32,shape=[batch_size,vocabulary_size]))
        
        
    #    
    #Encoder, Decoder and attention functions
    #
    
    def encoder(train_input,output,state):
        '''
        Args:       
        
        train_input : array of size num_unrolling, each array element is a Tensor of dimension batch_size,
        vocabulary size.
        
        Returns:
        
        output : Output of LSTM aka Hidden State
        state : Cell state of the LSTM
        
        '''
        i = len(train_inputs) - 1
        outputs=[]
        while i >= 0:
            output, state = encoder_lstm.lstm_cell(train_input[i],output,state)
            outputs.append(output)
            i=i-1
        #Return the all the outputs because they will be required by the attention mechanism
        return outputs,output,state
    
    def soft_attention(hidden_states,prev_hidden_state,batch_size):
        '''
        
        Implements soft attention mechanism over an array of encoder hidden 
        states given previous decoder hidden states
        
        Returns a context by attending over the hidden states accross time
        
        Used by the decoder at each timestep during decoding
        
        '''
        #Prev hidden weights
        prev_hidden_state_times_w=tf.multiply(prev_hidden_state,prev_hidden_weights)
        for h in range(num_unrollings):
            hidden_states[h]=tf.multiply(hidden_states[h],attn_weights)+prev_hidden_state_times_w 
        unrol_states=tf.reshape(tf.concat(hidden_states,1),(batch_size,num_unrollings,num_nodes))
        eij=tf.tanh(unrol_states)
        #Softmax across the unrolling dimension
        softmax=tf.nn.softmax(eij,dim=1)
        context=tf.reduce_sum(tf.multiply(softmax,unrol_states),axis=1) #Sum across axis time
        return context
        
    
    def training_decoder(decoder_input,hidden_states,output,state):
        outputs=[]
        #Predict the first character using the EOS Tag. We use EOS tag as the start tag
        context=soft_attention(hidden_states,output,batch_size)
        inp_concat=tf.concat([decoder_input[-1],context],axis=1)
        output, state = decoder_lstm.lstm_cell(inp_concat,output,state)
        outputs.append(output)
        #Now predict the next outputs using the training labels itself. Using y(n-1) to predict y(n)
        for i in decoder_input[0:-1]:
            context=soft_attention(hidden_states,output,batch_size)
            inp_concat=tf.concat([i,context],axis=1)
            output,state=decoder_lstm.lstm_cell(inp_concat,output,state)
            outputs.append(output)
            
        return outputs,output,state
    
    
    def inference_decoder(go_char,hidden_states,decode_steps,output,state):
        outputs=[]
        #First input to decoder is the the Go Character
        context=soft_attention(hidden_states,output,1)
        inp_concat=tf.concat([go_char,context],axis=1)
        output,state=decoder_lstm.lstm_cell(inp_concat,output,state)
        outputs.append(output)
        for i in range(decode_steps-1):
            #Feed the previous output as the next decoder input
            decoder_input=tf.nn.softmax(tf.nn.xw_plus_b(output, w, b))
            context=soft_attention(hidden_states,output,1)
            inp_concat=tf.concat([decoder_input,context],axis=1)
            output,state=decoder_lstm.lstm_cell(inp_concat,output,state)
            outputs.append(output)
        return outputs,output,state
    

        
    #
    #Model Definition
    #
    hidden_states,output,state=encoder(train_inputs,output,state)
    outputs,output,state=training_decoder(decoder_inputs,hidden_states,output,state)
    


    with tf.control_dependencies([saved_state.assign(state),
                                saved_output.assign(output),
                                    ]):
        logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), w, b)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                    labels=tf.concat(decoder_inputs, 0), logits=logits))
        
    #Loss function and optimizer
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
                            10.0, global_step, 5000, 0.1, staircase=True)
    optimizer = tf.train.AdamOptimizer()
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(
                    zip(gradients, v), global_step=global_step)
    
    # Predictions.
    train_prediction = tf.nn.softmax(logits)
    
    #  
    #Sample Prediction
    #
    sample_input=[]
    sample_outputs=[]

    for i in range(num_unrollings):
        sample_input.append(tf.placeholder(tf.float32,shape=[1,vocabulary_size]))
        
    sample_saved_state=tf.Variable(tf.zeros([1, num_nodes]), trainable=False)
    sample_saved_output=tf.Variable(tf.zeros([1, num_nodes]), trainable=False)
    

    
    sample_output=sample_saved_output
    sample_state=sample_saved_state

    
    
    reset_sample_state = tf.group(
        sample_output.assign(tf.zeros([1, num_nodes])),
        sample_state.assign(tf.zeros([1, num_nodes])),

        )
    

    hidden_states,sample_output,sample_state=encoder(sample_input,sample_output,sample_state)
    sample_decoder_outputs,sample_output,sample_state=inference_decoder(sample_input[-1],hidden_states,num_unrollings,sample_output,sample_state)

    with tf.control_dependencies([sample_saved_output.assign(sample_output),
                                sample_saved_state.assign(sample_state),
                               ]):
        for d in sample_decoder_outputs:
                sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(d, w, b))
                sample_outputs.append(sample_prediction)
    
        

num_steps = 20000
summary_frequency = 1000

with tf.Session(graph=graph) as session:
  global dropout
  tf.global_variables_initializer().run()
  print('Initialized')
  mean_loss = 0
  for step in range(num_steps):
    batches,output_batches = train_batches.next()
    feed_dict = dict()
    dropout=0.5
    
    for i in range(num_unrollings):
        #Feeding input from reverse according to https://arxiv.org/abs/1409.3215
        feed_dict[train_inputs[i]]=batches[i]
        feed_dict[decoder_inputs[i]]=output_batches[i]

        
    _, l, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    #reset_state.run()

    if step % (summary_frequency ) == 0:
        dropout=1
        print('-'*80)
        print('Step '+str(step))
        print('Loss '+str(l))
        
        labels=np.concatenate(list(output_batches)[:])
#        print(characters(labels))
#       print(characters(predictions))
        print('Batch Accuracy: %.2f' % float(accuracy(labels,predictions)*100))
        
        num_validation = valid_size // num_unrollings
        reset_sample_state.run()
        sum_acc=0
        for _ in range(num_validation):
            valid,valid_output=valid_batches.next()
            valid_feed_dict=dict()
            for i in range(num_unrollings):
                valid_feed_dict[sample_input[i]]=valid[i]
            sample_pred=session.run(sample_outputs,feed_dict=valid_feed_dict)
            labels=np.concatenate(list(valid_output)[:],axis=0)
            pred=np.concatenate(list(sample_pred)[:],axis=0)
            sum_acc=sum_acc+accuracy(labels,pred)
        val_acc=sum_acc/num_validation
        print('Validation Accuracy: %0.2f'%(val_acc*100))
        print('Input Test String '+str(batches2string(valid)))
        print('Output Prediction'+str(batches2string(sample_pred)))
        print('Actual'+str(batches2string(valid_output)))
        

