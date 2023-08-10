from dragonn import models

from sklearn.model_selection import train_test_split

from collections import OrderedDict
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

import sys

sys.path = ["/opt"] + sys.path

import dragonn

sys.path

key_to_seq = OrderedDict()

with open("../data/Pilot_counts_sequences/PilotDesign.sequences.txt") as f:
    for line in f:
        key, seq = line.strip().split()
        key_to_seq[key] = seq
        
pprint(key_to_seq.items()[:5])

key_to_normalized_K562_Rep1 = {}
key_to_normalized_K562_Rep2 = {}

with open("../data/Pilot_normalized/K562/tablenorm_recenterends_K562_Rep1_20.txt") as f:
    for line in f:
        parts = line.strip().split()
        
        for i, norm in enumerate(parts[1:]):
            key = "{}_{}".format(parts[0], i)
            val = float(norm)
            key_to_normalized_K562_Rep1[key] = val
                
with open("../data/Pilot_normalized/K562/tablenorm_recenterends_K562_Rep2_20.txt") as f:
    for line in f:
        parts = line.strip().split()
        
        for i, norm in enumerate(parts[1:]):
            key = "{}_{}".format(parts[0], i)
            val = float(norm)
            key_to_normalized_K562_Rep2[key] = val

# Check that the sequence and value keys line up.
assert set(key_to_normalized_K562_Rep1.keys()) == set(key_to_seq.keys())
assert set(key_to_normalized_K562_Rep2.keys()) == set(key_to_seq.keys())

# One hot encode DNA sequences the standard way.
def one_hot_encode_seq(seq):
    bases = ['A', 'T', 'C', 'G']
    # Gotta be ready for when we discover a new base.
    result = np.zeros((4, len(seq)))
    
    for i, base in enumerate(seq):
        result[bases.index(base), i] = 1
    return result

def seqs_to_encoded_matrix(seqs):
    # Wrangle the data into a shape that Dragonn wants.
    result = np.concatenate(
        map(one_hot_encode_seq, seqs)
    ).reshape(
        len(seqs), 1, 4, len(seqs[0])
    )
    
    # Check we actually did the encoding right.
    for i in range(len(seqs)):
        for j in range(len(seqs[0])):
            assert sum(result[i, 0, :, j]) == 1
    
    return result

X = seqs_to_encoded_matrix(key_to_seq.values())

# Just round to the median, to make this a classification task for now.
K562_Rep1_median = np.median(key_to_normalized_K562_Rep1.values())
K562_Rep1_y = np.array(
    map(
        lambda key: key_to_normalized_K562_Rep1[key] > K562_Rep1_median, 
        key_to_seq.keys()
    )
).reshape(-1, 1)

K562_Rep2_median = np.median(key_to_normalized_K562_Rep2.values())
K562_Rep2_y = np.array(
    map(
        lambda key: key_to_normalized_K562_Rep2[key] > K562_Rep2_median, 
        key_to_seq.keys()
    )
).reshape(-1, 1)

y = np.hstack([K562_Rep1_y, K562_Rep2_y])

get_ipython().magic('pinfo train_test_split')

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.1, random_state=42
)

model = models.SequenceDNN(
    seq_length=X_train.shape[3],
    num_filters=[100, 10],
    conv_width=[10, 10],
    num_tasks=y_train.shape[1]
)

model.train(X_train, y_train, (X_valid, y_valid))

def print_loss(model):
    train_losses, valid_losses = [np.array([epoch_metrics['Loss'] for epoch_metrics in metrics])
                                  for metrics in (model.train_metrics, model.valid_metrics)]

    # Pretty sure early stopping works by taking the mean of losses, might want to double check
    train_losses = train_losses.mean(axis=1)
    valid_losses = valid_losses.mean(axis=1)

    f = plt.figure(figsize=(10, 4))
    ax = f.add_subplot(1, 1, 1)
    
    ax.plot(range(len(train_losses)), train_losses, label='Training',lw=4)
    ax.plot(range(len(train_losses)), valid_losses, label='Validation', lw=4)
    
    min_loss_indx = min(enumerate(valid_losses), key=lambda x: x[1])[0]
    ax.plot([min_loss_indx, min_loss_indx], [0, 1.0], 'k--', label='Early Stop')
    ax.legend(loc="upper right")
    ax.set_ylabel("Loss")
    ax.set_ylim((0.0,1.0))
    ax.set_xlabel("Epoch")
    plt.show()

print_loss(model)

multi_filter_model = models.SequenceDNN(
    seq_length=X_train.shape[3],
    num_filters=[15],
    conv_width=[45],
    pool_width=45,
    dropout=0.1,
    num_tasks=y_train.shape[1]
)

multi_filter_model.train(X_train, y_train, (X_valid, y_valid))

print_loss(multi_filter_model)



