get_ipython().magic('matplotlib inline')
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns

import torch
from torch import FloatTensor
from torch.autograd import Variable
from torch.nn import functional as F, Linear

#
# Helper functions
#

def inner_prod(x, y):
    """Compute the inner products between rows in x and y."""
    return (x * y).sum(1)


def cosine_sim(x, y):
    """Compute the cosine similarity between row of x and y."""
    Z = x.norm(dim=1) * y.norm(dim=1)
    return inner_prod(x, y) / Z


def all_same_size(seq):
    """Check that all elements of seq have the same size."""
    if len(seq) < 1:
        return True
    size = seq[0].size()
    for x in seq:
        if x.size() != size:
            return False
    return True

#
# Attention
#

def weighted_sum(weights, values):
    """Compute the sum of weights[i] * values[i].

    Expands weights[i] to match values[i].

    Raises:
        ValueError: if len(weights) != len(values)
        ValueError: if len(weights) == len(values) == 0
        ValueError: if weights[i].expand_as(values[i]) fails
        RuntimeError: if all weights[i] * values[i] can not be summed 
    """
    if len(weights) != len(values):
        raise ValueError('len(weights) = {} != {} = len(values)'.format(len(weights), len(values)))
    if len(weights) == 0:
        raise ValueError('cannot computed weighted sum of empty sequences')
    
    return sum(w.expand_as(v) * v for w, v in zip(weights, values))


def attend(query, keys, values=None, score=inner_prod):
    """Apply softmax attention over keys.

    Args:
        query: the query to compare to each key.
        keys: an iterable of items to compare to query.
        values: if given returns the weighted sum of probs[i] * values[i].
        score: the function used to compare each (key, query) pair.

    Returns:
        probs if values is None, otherwise the weighted sum of probs[i] * values[i].
    
    Raises:
        ValueError: if len(keys) < 1
        See weighted_sum for other potential errors.
    """
    n_keys = len(keys)
    if n_keys < 1:
        raise ValueError('cannot compute attention over empty sequence')
    
    scores = [score(key, query) for key in keys]
    probs = F.softmax(torch.cat(scores, 1)).chunk(n_keys, 1)
    
    if values is None:
        return probs
    return weighted_sum(probs, values)

#
# Unit tests
#

def test():
    import unittest
    
    def FloatVar(x):
        return Variable(FloatTensor(x))

    class AttentionTest(unittest.TestCase):
        batch_size = 2
        n_mem = 3
        k_dim = 4
        v_dim = 5


        Q = [[ 5,  4, -1,  2],
             [ 5,  4,  2,  1]]

        K = [[[ 5, -5, -1, -2],
              [ 1,  0,  5,  0],
              [ 1, -3, -4,  1]],
             [[-1,  3,  5,  0],
              [-1, -4,  4,  1],
              [ 0,  0, -1,  4]]]

        V = [[[ 5,  2, -1,  0, -5],
              [ 1,  4, -1, -5, -5],
              [-4, -5,  0, -3,  3]],
             [[ 3,  5,  1, -3, -3],
              [-4, -2,  4,  3, -5],
              [ 2, -4, -1,  0,  4]]]

        p_q0_K0 = [0.8437947344813395, 0.11419519938459449, 0.04201006613406605]
        sum_p_q0_K0_K0 = [4.375178937925358, -4.345003870808895, -0.44085900209463125, -1.6455794028286128]
        sum_p_q0_K0_V0 = [4.165128607255028, 1.9343199358307268, -0.9579899338659339, -0.6970061953251705, -4.663919470927472]

        p_q1_K0 = [8.315280276639204e-07, 0.9999991684717178, 2.543663532246996e-13]
        sum_p_q1_K0_K0 = [1.0000033261121106, -4.157640901418662e-06, 4.999995010829543, -1.6630558009614876e-06]
        sum_p_q1_K0_V0 = [1.0000033261108388, 3.9999983369416547, -0.9999999999997454, -4.999995842359351, -4.999999999997963]

        p_q1_K1 = [0.9999996940975185, 2.5436648692632894e-13, 3.0590222692554685e-07]
        sum_p_q1_K1_K1 = [-0.9999996940977729, 2.9999990822915383, 4.999998164586383, 1.2236091620686743e-06]
        sum_p_q1_K1_V1 = [2.999999694095992, 4.999997246878176, 0.9999993881963092, -2.999999082291793, -2.99999785868492]
        
        def test_1xn_1xn(self):
            q = FloatVar([self.Q[0]])
            ks = [FloatVar([v]) for v in self.K[0]]
            vs = [FloatVar([k]) for k in self.V[0]]
            
            p = np.hstack(p.data.numpy() for p in attend(q, ks))
            self.assertEqual(p.shape, (1, self.n_mem))
            self.assertTrue(np.allclose(p, np.array([self.p_q0_K0])))

            h = attend(q, ks, ks).data.numpy()
            self.assertEqual(h.shape, (1, self.k_dim))
            self.assertTrue(np.allclose(h, np.array([self.sum_p_q0_K0_K0])))

            g = attend(q, ks, vs).data.numpy()
            self.assertEqual(g.shape, (1, self.v_dim))
            self.assertTrue(np.allclose(g, np.array([self.sum_p_q0_K0_V0])))
            
        def test_mxn_1xn(self):
            q = FloatVar(self.Q)
            ks = [FloatVar([k]).expand(self.batch_size, self.k_dim) for k in self.K[0]]
            vs = [FloatVar([v]).expand(self.batch_size, self.v_dim) for v in self.V[0]]
            
            p = np.hstack(p.data.numpy() for p in attend(q, ks))
            self.assertEqual(p.shape, (self.batch_size, self.n_mem))
            self.assertTrue(np.allclose(p, np.array([self.p_q0_K0, self.p_q1_K0])))

            h = attend(q, ks, ks).data.numpy()
            self.assertEqual(h.shape, (self.batch_size, self.k_dim))
            self.assertTrue(np.allclose(h, np.array([self.sum_p_q0_K0_K0, self.sum_p_q1_K0_K0])))

            g = attend(q, ks, vs).data.numpy()
            self.assertEqual(g.shape, (self.batch_size, self.v_dim))
            self.assertTrue(np.allclose(g, np.array([self.sum_p_q0_K0_V0, self.sum_p_q1_K0_V0])))

        def test_mxn_mxn(self):
            q = FloatVar(self.Q)
            ks = [FloatVar(k) for k in zip(*self.K)]
            vs = [FloatVar(v) for v in zip(*self.V)]
            
            p = np.hstack(p.data.numpy() for p in attend(q, ks))
            self.assertEqual(p.shape, (self.batch_size, self.n_mem))
            self.assertTrue(np.allclose(p, np.array([self.p_q0_K0, self.p_q1_K1])))

            h = attend(q, ks, ks).data.numpy()
            self.assertEqual(h.shape, (self.batch_size, self.k_dim))
            self.assertTrue(np.allclose(h, np.array([self.sum_p_q0_K0_K0, self.sum_p_q1_K1_K1])))

            g = attend(q, ks, vs).data.numpy()
            self.assertEqual(g.shape, (self.batch_size, self.v_dim))
            self.assertTrue(np.allclose(g, np.array([self.sum_p_q0_K0_V0, self.sum_p_q1_K1_V1])))

        def test_things_that_should_fail(self):
            with self.assertRaises(ValueError):
                # ks is empty
                q = FloatVar(self.Q)
                attend(q, [])

            with self.assertRaises(ValueError):
                # vs is empty
                q = FloatVar(self.Q)
                ks = [FloatVar(np.random.random((self.batch_size, self.k_dim))) for i in range(2)]
                attend(q, ks, [])

            with self.assertRaises(ValueError):
                # len(ks) != len(vs)
                q = FloatVar(self.Q)
                ks = [FloatVar(np.random.random((self.batch_size, self.k_dim))) for i in range(2)]
                vs = [FloatVar(np.random.random((self.batch_size, self.v_dim)))]
                attend(q, ks, vs)

            with self.assertRaises(RuntimeError):
                # ks[i] has wrong batch_size, inner_prod fails
                q = FloatVar(self.Q)
                ks = [FloatVar(np.random.random((1, self.k_dim))) for i in range(2)]
                attend(q, ks)

            with self.assertRaises(RuntimeError):
                # ks[i] has wrong batch_size, inner_prod fails
                q = FloatVar(self.Q)
                ks = [FloatVar(np.random.random((self.batch_size + 1, self.k_dim))) for i in range(2)]
                attend(q, ks)

            with self.assertRaises(ValueError):
                # vs[i] has wrong batch_size, expand_as fails
                q = FloatVar(self.Q)
                ks = [FloatVar(np.random.random((self.batch_size, self.k_dim))) for i in range(2)]
                vs = [FloatVar(np.random.random((self.batch_size + 1, self.v_dim))) for i in range(2)]
                attend(q, ks, vs)
                
            with self.assertRaises(RuntimeError):
                # vs[0].size() != vs[1].size(), sum fails
                q = FloatVar(self.Q)
                ks = [FloatVar(np.random.random((self.batch_size, self.k_dim))) for i in range(2)]
                vs = [FloatVar(np.random.random((self.batch_size, self.v_dim))),
                      FloatVar(np.random.random((self.batch_size, self.v_dim + 1)))]
                attend(q, ks, vs)
            

    unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(AttentionTest))

test()

#
# Create training data
#


min_dim, max_dim = 3, 7
input_dim = 10
samples_per_length = 10
lengths = [5, 10, 15, 20]
q_min = np.random.normal(0, 1, input_dim).tolist()
q_max = np.random.normal(0, 1, input_dim).tolist()

mini_batches = []
for length in lengths:
    sequences = [[] for i in range(length)]
    queries = []
    targets = []
    for i in range(samples_per_length):
        seq = np.random.normal(0, 1, (length, input_dim)).tolist()
        min_target = min(seq, key=lambda x: x[min_dim])
        max_target = max(seq, key=lambda x: x[max_dim])
        for t, x in enumerate(seq):
            sequences[t].extend([x, x])
        queries.extend([q_min, q_max])
        targets.extend([min_target, max_target])

    sequences = [Variable(FloatTensor(x)) for x in sequences]
    queries = Variable(FloatTensor(queries))
    targets = Variable(FloatTensor(targets))
    mini_batches.append((sequences, queries, targets))

#
# Train
# 

embed_dim = 5
w = Linear(input_dim, embed_dim)
u = Linear(input_dim, embed_dim)
params = list(w.parameters()) + list(u.parameters())
opt = torch.optim.SGD(params, lr=0.01)
loss = torch.nn.MSELoss()

epoch = 0
prev_err = float('inf')
patience = 100
frustration = 0
tolerance = 1e-5
loss_values = []
while True:
    epoch += 1
    sum_err = 0.0
    random.shuffle(mini_batches)
    for xs, q, y in mini_batches:
        opt.zero_grad()
        p = attend(u(q), [w(x) for x in xs], xs)
        err = loss(p, y)
        err.backward()
        opt.step()
        sum_err += err.data.numpy()[0]
    err = sum_err / len(mini_batches)
    loss_values.append(err)
#     if epoch % 50 == 0:
#         print('epoch[{}] err = {}'.format(epoch, err))
    if err < prev_err - tolerance:
        frustration = 0
    else:
        frustration += 1
    if frustration > patience:
        break
    prev_err = err

plt.plot(loss_values)
plt.xlim(0, len(loss_values))
plt.xlabel('epoch')
plt.ylabel('mse')

#
# Test
#


n_test = 10
errors_min = 0
errors_max = 0
length = 50

min_data, max_data = [], []

for i in range(n_test):
    seq = np.random.normal(0, 1, (length, input_dim)).tolist()
    
    min_values = [x[min_dim] for x in seq]
    min_sort = np.argsort([-x for x in min_values])
    min_index = min_sort[-1]
    
    max_values = [x[max_dim] for x in seq]
    max_sort = np.argsort(max_values)
    max_index = max_sort[-1]

    xs = [Variable(FloatTensor([x]), volatile=True) for x in seq]
    q = Variable(FloatTensor([q_min]), volatile=True)
    min_probs = np.array([a.data.numpy()[0,0] for a in attend(u(q), [w(x) for x in xs])])
    min_index_pred = min_probs.argmax()
    if min_index != min_index_pred:
        errors_min += 1

    q = Variable(FloatTensor([q_max]), volatile=True)
    max_probs = np.array([a.data.numpy()[0,0] for a in attend(u(q), [w(x) for x in xs])])
    max_index_pred = max_probs.argmax()
    if max_index != max_index_pred:
        errors_max += 1
        
    min_data.append((min_values, min_probs, min_index))
    max_data.append((max_values, max_probs, max_index))

print('errors min: {} of {}'.format(errors_min, n_test))
print('errors max: {} of {}'.format(errors_max, n_test))
print('error: {:%}'.format((errors_min + errors_max) / float(2 * n_test)))

colors = sns.color_palette()
fig, axs = plt.subplots(len(min_data), 2, figsize=(10, 20), sharex=True)
axs_min, axs_max = axs[:,0], axs[:,1]

for i, ((min_values, min_probs, min_index), (max_values, max_probs, max_index)) in enumerate(zip(min_data, max_data)):
    
    min_lims = min(min_values) - 1, max(min_values) + 1
    axs_min[i].bar(np.arange(len(min_sort)) - 0.4, min_values, color=colors[3], zorder=2, label='values')
    axs_min[i].plot(min_probs, color=colors[2], lw=3, zorder=3, label='attention')
    axs_min[i].bar([min_index - 0.4], [min_lims[1] - min_lims[0]], bottom=[min_lims[0]], color=colors[1])
    axs_min[i].set_xlim(-0.4, len(min_sort) - 0.6)
    axs_min[i].set_ylim(min_lims)

    max_lims = min(max_values) - 1, max(max_values) + 1
    axs_max[i].bar(np.arange(len(max_sort)) - 0.4, max_values, color=colors[3], zorder=2, label='values')
    axs_max[i].plot(max_probs, color=colors[2], lw=3, zorder=3, label='attention')
    axs_max[i].bar([max_index - 0.4], [max_lims[1] - max_lims[0]], bottom=[max_lims[0]], color=colors[1])
    axs_max[i].set_xlim(-0.4, len(max_sort) - 0.6)
    axs_max[i].set_ylim(max_lims)

axs_min[0].set_title('Minimum')
axs_max[0].set_title('Maximum')
axs_max[0].legend(loc='best')    
axs_min[0].legend(loc='best')
axs_max[0].legend(loc='best')
axs_min[-1].set_xlabel('position')
axs_max[-1].set_xlabel('position')
plt.tight_layout()



