from __future__ import print_function
import requests
import os

def download(url, filename):
    """ utility function to download a file """
    response = requests.get(url, stream=True)
    with open(filename, "wb") as handle:
        for data in response.iter_content():
            handle.write(data)

locations = ['Tutorials/SLUHandsOn', 'Examples/LanguageUnderstanding/ATIS/BrainScript']

data = {
  'train': { 'file': 'atis.train.ctf', 'location': 0 },
  'test': { 'file': 'atis.test.ctf', 'location': 0 },
  'query': { 'file': 'query.wl', 'location': 1 },
  'slots': { 'file': 'slots.wl', 'location': 1 }
}

for item in data.values():
    location = locations[item['location']]
    path = os.path.join('..', location, item['file'])
    if os.path.exists(path):
        print("Reusing locally cached:", item['file'])
        # Update path
        item['file'] = path
    elif os.path.exists(item['file']):
        print("Reusing locally cached:", item['file'])
    else:
        print("Starting download:", item['file'])
        url = "https://github.com/Microsoft/CNTK/blob/v2.0.rc2/%s/%s?raw=true"%(location, item['file'])
        download(url, item['file'])
        print("Download completed")

import math
import numpy as np

import cntk as C
from cntk.logging import ProgressPrinter, log_number_of_parameters
from cntk.io import MinibatchSource, CTFDeserializer
from cntk.io import StreamDef, StreamDefs, INFINITELY_REPEAT
from cntk import *
from cntk.learners import fsadagrad, learning_rate_schedule
from cntk.layers import * # CNTK Layers library


# Select the right target device when this notebook is being tested:
C.device.try_set_default_device(C.device.gpu(0))

# number of words in vocab, slot labels, and intent labels
vocab_size = 943 ; num_labels = 129 ; num_intents = 26    

# model dimensions
input_dim  = vocab_size
label_dim  = num_labels
emb_dim    = 150
hidden_dim = 300

def create_model():
    with default_options(initial_state=0.1):
        return Sequential([
            Embedding(emb_dim, name='embed'),
            Recurrence(LSTM(hidden_dim), go_backwards=False),
            Dense(num_labels, name='classify')
        ])

# peek
model = create_model()
print(model.embed.E.shape)
print(model.classify.b.value)

def create_reader(path, is_training):
    return MinibatchSource(CTFDeserializer(path, StreamDefs(
         query         = StreamDef(field='S0', shape=vocab_size,  is_sparse=True),
         intent_unused = StreamDef(field='S1', shape=num_intents, is_sparse=True),  
         slot_labels   = StreamDef(field='S2', shape=num_labels,  is_sparse=True)
     )), randomize=is_training, max_sweeps = INFINITELY_REPEAT if is_training else 1)

# peek
reader = create_reader(data['train']['file'], is_training=True)
reader.streams.keys()

def create_criterion_function(model):
    labels = C.placeholder(name='labels')
    ce   = cross_entropy_with_softmax(model, labels)
    errs = classification_error      (model, labels)
    return combine ([ce, errs]) # (features, labels) -> (loss, metric)

def train(reader, model, max_epochs=16):
    # criterion: (model args, labels) -> (loss, metric)
    #   here  (query, slot_labels) -> (ce, errs)
    criterion = create_criterion_function(model)

    criterion.replace_placeholders({criterion.placeholders[0]: C.sequence.input(vocab_size),
                                    criterion.placeholders[1]: C.sequence.input(num_labels)})

    # training config
    epoch_size = 18000        # 18000 samples is half the dataset size 
    minibatch_size = 70
    
    # LR schedule over epochs 
    # In CNTK, an epoch is how often we get out of the minibatch loop to
    # do other stuff (e.g. checkpointing, adjust learning rate, etc.)
    # (we don't run this many epochs, but if we did, these are good values)
    lr_per_sample = [0.003]*4+[0.0015]*24+[0.0003]
    lr_per_minibatch = [x * minibatch_size for x in lr_per_sample]
    lr_schedule = learning_rate_schedule(lr_per_minibatch, UnitType.minibatch, epoch_size)
    
    # Momentum
    momentum_as_time_constant = momentum_as_time_constant_schedule(700)
    
    # We use a variant of the FSAdaGrad optimizer which is known to work well on this dataset
    # Feel free to try other optimizers from 
    # https://www.cntk.ai/pythondocs/cntk.learner.html#module-cntk.learner
    learner = fsadagrad(criterion.parameters,
                        lr=lr_schedule, momentum=momentum_as_time_constant,
                        gradient_clipping_threshold_per_sample=15, gradient_clipping_with_truncation=True)

    # trainer
    progress_printer = ProgressPrinter(tag='Training', num_epochs=max_epochs)
    #progress_printer = ProgressPrinter(freq=100, first=10, tag='Training', num_epochs=max_epochs) # more detailed logging
    trainer = Trainer(model, criterion, learner, progress_printer)

    # process minibatches and perform model training
    log_number_of_parameters(model)

    t = 0
    for epoch in range(max_epochs):         # loop over epochs
        epoch_end = (epoch+1) * epoch_size
        while t < epoch_end:                # loop over minibatches on the epoch
            data = reader.next_minibatch(minibatch_size, input_map={  # fetch minibatch
                criterion.arguments[0]: reader.streams.query,
                criterion.arguments[1]: reader.streams.slot_labels
            })
            trainer.train_minibatch(data)                                     # update model with it
            t += data[criterion.arguments[1]].num_samples                     # samples so far
        trainer.summarize_training_progress()

def do_train():
    global model
    model = create_model()
    reader = create_reader(data['train']['file'], is_training=True)
    train(reader, model)
do_train()

def evaluate(reader, model):
    criterion = create_criterion_function(model)
    criterion.replace_placeholders({criterion.placeholders[0]: C.sequence.input(num_labels)})

    # process minibatches and perform evaluation
    lr_schedule = learning_rate_schedule(1, UnitType.minibatch)
    momentum_as_time_constant = momentum_as_time_constant_schedule(0)
    dummy_learner = fsadagrad(criterion.parameters, 
                              lr=lr_schedule, momentum=momentum_as_time_constant)
    progress_printer = ProgressPrinter(tag='Evaluation', num_epochs=0)
    evaluator = Trainer(model, criterion, dummy_learner, progress_printer)

    while True:
        minibatch_size = 500
        data = reader.next_minibatch(minibatch_size, input_map={  # fetch minibatch
            criterion.arguments[0]: reader.streams.query,
            criterion.arguments[1]: reader.streams.slot_labels
        })
        if not data:                                 # until we hit the end
            break
        evaluator.test_minibatch(data)
    evaluator.summarize_test_progress()

def do_test():
    reader = create_reader(data['test']['file'], is_training=False)
    evaluate(reader, model)
do_test()
model.classify.b.value

# load dictionaries
query_wl = [line.rstrip('\n') for line in open(data['query']['file'])]
slots_wl = [line.rstrip('\n') for line in open(data['slots']['file'])]
query_dict = {query_wl[i]:i for i in range(len(query_wl))}
slots_dict = {slots_wl[i]:i for i in range(len(slots_wl))}

# let's run a sequence through
seq = 'BOS flights from new york to seattle EOS'
w = [query_dict[w] for w in seq.split()] # convert to word indices
print(w)
onehot = np.zeros([len(w),len(query_dict)], np.float32)
for t in range(len(w)):
    onehot[t,w[t]] = 1
pred = model.eval({model.arguments[0]:[onehot]})[0]
print(pred.shape)
best = np.argmax(pred,axis=1)
print(best)
list(zip(seq.split(),[slots_wl[s] for s in best]))

# Your task: Add batch normalization
def create_model():
    with default_options(initial_state=0.1):
        return Sequential([
            Embedding(emb_dim),
            Recurrence(LSTM(hidden_dim), go_backwards=False),
            Dense(num_labels)
        ])

# Enable these when done:
#do_train()
#do_test()

# Your task: Add lookahead
def create_model():
    with default_options(initial_state=0.1):
        return Sequential([
            Embedding(emb_dim),
            Recurrence(LSTM(hidden_dim), go_backwards=False),
            Dense(num_labels)
        ])
    
# Enable these when done:
#do_train()
#do_test()

# Your task: Add bidirectional recurrence
def create_model():
    with default_options(initial_state=0.1):  
        return Sequential([
            Embedding(emb_dim),
            Recurrence(LSTM(hidden_dim), go_backwards=False),
            Dense(num_labels)
        ])

# Enable these when done:
#do_train()
#do_test()

def create_model():
    with default_options(initial_state=0.1):
        return Sequential([
            Embedding(emb_dim),
            BatchNormalization(),
            Recurrence(LSTM(hidden_dim), go_backwards=False),
            BatchNormalization(),
            Dense(num_labels)
        ])

do_train()
do_test()

def OneWordLookahead():
    x = C.placeholder()
    apply_x = splice (x, sequence.future_value(x))
    return apply_x

def create_model():
    with default_options(initial_state=0.1):
        return Sequential([
            Embedding(emb_dim),
            OneWordLookahead(),
            Recurrence(LSTM(hidden_dim), go_backwards=False),
            Dense(num_labels)        
        ])

do_train()
do_test()

def BiRecurrence(fwd, bwd):
    F = Recurrence(fwd)
    G = Recurrence(bwd, go_backwards=True)
    x = C.placeholder()
    apply_x = splice (F(x), G(x))
    return apply_x 

def create_model():
    with default_options(initial_state=0.1):
        return Sequential([
            Embedding(emb_dim),
            BiRecurrence(LSTM(hidden_dim//2), LSTM(hidden_dim//2)),
            Dense(num_labels)
        ])

do_train()
do_test()



