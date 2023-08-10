get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import datajoint as dj
from database import LnpFit, Net, Fit, NetFC, FitFC, NetFixedMask, FitFixedMask
from collections import OrderedDict
import tensorflow as tf

def fetch_best(rel, *args):
    results = rel.fetch(*args, order_by='val_loss', limit=1)
    return [r[0] for r in results]

def get_n_layer_nets(region_num, num_layers):
    return list(Net().aggregate(
        Net.ConvLayer(), num_layers='count(*)').restrict(
        dict(region_num=region_num, num_layers=num_layers)).fetch(dj.key))

num_neurons = [103, 55, 102]
test_corrs = OrderedDict((
    ('Antolik', [0.51, 0.43, 0.46]),
    ('LNP', []),
    ('CNN 1 layer', []),
    ('CNN 2 layers', []),
    ('CNN 3 layers', []),
    ('CNN fully-connected readout', []),
    ('CNN fixed mask', []),
))
val_loss = OrderedDict((
    ('Antolik', [0, 0, 0]),
    ('LNP', []),
    ('CNN 1 layer', []),
    ('CNN 2 layers', []),
    ('CNN 3 layers', []),
    ('CNN fully-connected readout', []),
    ('CNN fixed mask', []),
))
best_net_key = OrderedDict((
    ('Antolik', []),
    ('LNP', []),
    ('CNN 1 layer', []),
    ('CNN 2 layers', []),
    ('CNN 3 layers', []),
    ('CNN fully-connected readout', []),
    ('CNN fixed mask', []),
))

for region_num in range(1, 4):
    region_key = {'region_num': region_num}
    r, l, k = fetch_best(LnpFit() & region_key, 'avg_corr', 'val_loss', dj.key)
    test_corrs['LNP'].append(r)
    val_loss['LNP'].append(l)
    best_net_key['LNP'].append(k)
    for n in range(1, 4):
        keys = get_n_layer_nets(region_num, num_layers=n)
        r, l, k = fetch_best(Fit() & region_key & keys, 'avg_corr', 'val_loss', dj.key)
        cnn = 'CNN {:d} layer'.format(n) + ('s' if n > 1 else '')
        test_corrs[cnn].append(r)
        val_loss[cnn].append(l)
        best_net_key[cnn].append(k)

    r, l, k = fetch_best(FitFC() & region_key, 'avg_corr', 'val_loss', dj.key)
    test_corrs['CNN fully-connected readout'].append(r)
    val_loss['CNN fully-connected readout'].append(l)
    best_net_key['CNN fully-connected readout'].append(k)
    r, l, k = fetch_best(FitFixedMask() & region_key, 'avg_corr', 'val_loss', dj.key)
    test_corrs['CNN fixed mask'].append(r)
    val_loss['CNN fixed mask'].append(l)
    best_net_key['CNN fixed mask'].append(k)

def results_table(results, n=None):
    print_avg = (n is not None)
    row = '{:30s}'.format('Region')
    for i in range(3):
        row += '  {:5d}'.format(i+1)
    if print_avg:
        row += '    Avg'
    print(row)
    print((58 if print_avg else 51) * '-')
    for model, val in results.items():
        row = '{:30s}'.format(model)
        for v in val:
            row += '  {:5.2f}'.format(v)
        if print_avg:
            avg = np.sum(np.array(val) * np.array(n)) / np.sum(n)
            row += '  {:5.2f}'.format(avg)
        print(row)
    print(' ')

print('Average correlations on test set')
results_table(test_corrs, num_neurons)

print('Loss on validation set')
results_table(val_loss)

for i, k in enumerate(best_net_key['CNN 3 layers']):
    print('Region {:d}'.format(i+1))
    print(Net.ConvLayer() & k)

for i, k in enumerate(best_net_key['CNN fixed mask']):
    print('Region {:d}'.format(i+1))
    print(NetFixedMask.ConvLayer() & k)

for i, k in enumerate(best_net_key['CNN fully-connected readout']):
    print('Region {:d}'.format(i+1))
    print(NetFC.ConvLayer() & k)

