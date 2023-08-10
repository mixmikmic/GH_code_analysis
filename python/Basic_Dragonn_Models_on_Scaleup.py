from dragonn import models
from dragonn.plot import add_letters_to_axis

from sklearn.model_selection import train_test_split

from collections import OrderedDict
from pprint import pprint
from warnings import warn

import numpy as np
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

key_to_seq = OrderedDict()

with open("../data/Scaleup_counts_sequences/ScaleUpDesign1.sequences.txt") as f:
    for line in f:
        key, seq = line.strip().split()
        
        # TODO: Figure out if this is an OK thing to do. 'N' basically means the 
        # sequencing software couldn't figure out what the base was...?
        if "N" in seq:
            warn("Replacing 'N' bases in seq with 'A' in seq {}.".format(seq))
            seq = seq.replace("N", "A")
        
        assert key not in key_to_seq
        key_to_seq[key] = seq
        
with open("../data/Scaleup_counts_sequences/ScaleUpDesign2.sequences.txt") as f:
    for line in f:
        key, seq = line.strip().split()
        
        if "N" in seq:
            warn("Replacing 'N' bases in seq with 'A' in seq {}.".format(seq))
            seq = seq.replace("N", "A")
        
        assert key not in key_to_seq
        key_to_seq[key] = seq
        
pprint(key_to_seq.items()[:5])

print "{} total sequences of length {}".format(len(key_to_seq), len(key_to_seq.values()[0]))

data = {}
cell_types =  ["HepG2", "K562"]
promoters = ["SV40P", "minP"]
design_names = ["ScaleUpDesign1", "ScaleUpDesign2"]

for cell_type in cell_types:
    for promoter in promoters:
        experiment_key = (cell_type, promoter)

        for design_name in design_names:
            data[experiment_key] = {}

            with open("../data/Scaleup_normalized/{}_{}_{}_mRNA_Rep1.normalized".format(cell_type, design_name, promoter)) as f:
                for line in f:
                    parts = line.strip().split()

                    key = parts[0]
                    val = float(parts[1])
                    if parts[2] == "1":
                        data[experiment_key][key] = val

            with open("../data/Scaleup_normalized/{}_{}_{}_mRNA_Rep2.normalized".format(cell_type, design_name, promoter)) as f:
                for line in f:
                    parts = line.strip().split()

                    key = parts[0]
                    val = float(parts[1])
                    if parts[2] == "1" and key in data[experiment_key]:
                        data[experiment_key][key] = (val + data[experiment_key][key]) / 2.0
            
print "Data from experiment {}:".format(data.items()[0][0])
pprint(data.items()[0][1].items()[:5])

for key, val in data.items():
    print "Experiment {} has {} measurements.".format(key, len(val))

# One hot encode DNA sequences the standard way.
bases = ['A', 'T', 'C', 'G']

def one_hot_encode_seq(seq):
    result = np.zeros((len(bases), len(seq)))
    
    for i, base in enumerate(seq):
        result[bases.index(base), i] = 1

    return result

def seqs_to_encoded_matrix(seqs):
    # Wrangle the data into a shape that Dragonn wants.
    result = np.concatenate(
        map(one_hot_encode_seq, seqs)
    ).reshape(
        len(seqs), 1, len(bases), len(seqs[0])
    )
    
    # Check we actually did the encoding right.
    for i in range(len(seqs)):
        for j in range(len(seqs[0])):
            assert sum(result[i, 0, :, j]) == 1
    
    return result

valid_keys = list(reduce(
    lambda acc, d: acc.intersection(d.keys()), 
    data.values()[1:], 
    set(data.values()[0].keys())
))

print "{} sequences have measurements for all tasks.".format(len(valid_keys))

X = seqs_to_encoded_matrix([key_to_seq[key] for key in valid_keys])

X.shape

# Just round to the median, to make this a classification task for now.
experiment_labels = []
for experiment_key, key_to_normalized in data.items():
    
    filtered_normalized = [key_to_normalized[key] for key in valid_keys]
    
    median = np.median(filtered_normalized)
    experiment_labels.append(
        np.array(map(lambda val: val > median, filtered_normalized)).reshape(-1, 1)
    )

y = np.hstack(experiment_labels)

y.shape

X[:5, :, :, :]

y[:5, :]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = models.SequenceDNN(
    seq_length=X_train.shape[3],
    num_filters=[10],
    conv_width=[15],
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

def plot_sequence_filters(dnn):
    fig = plt.figure(figsize=(15, 8))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    conv_filters = dnn.get_sequence_filters()
    num_plots_per_axis = int(len(conv_filters)**0.5) + 1
    for i, conv_filter in enumerate(conv_filters):
        ax = fig.add_subplot(num_plots_per_axis, num_plots_per_axis, i+1)
        add_letters_to_axis(ax, conv_filter.T)
        ax.axis("off")
        ax.set_title("Filter %s" % (str(i+1)))

plot_sequence_filters(model)

multi_filter_model = models.SequenceDNN(
    seq_length=X_train.shape[3],
    num_filters=[15, 15, 15],
    conv_width=[15, 15, 15],
    num_tasks=y_train.shape[1],
    dropout=0.1
)

multi_filter_model.train(X_train, y_train, (X_valid, y_valid))

print_loss(multi_filter_model)



