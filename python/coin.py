get_ipython().magic('matplotlib inline')
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, Normalize
from tempfile import NamedTemporaryFile
import seaborn as sns
import numpy as np
import six
import tensorflow as tf

plt.style.use('seaborn-talk')
sns.set_context("talk", font_scale=1.4)
sess = ed.get_session()

sns.palplot(sns.color_palette())

# this can be done only before using Edward
ed.set_seed(42)

from edward.models import Bernoulli, Beta, Empirical, Uniform

N = 100  # number of coin flip observations in dataset

def build_fair_dataset(N):
    pheads = tf.constant(0.5)
    c = Bernoulli(probs=pheads, sample_shape=N)
    return sess.run([pheads, c])

def build_unfair_dataset(N):
    pheads = tf.constant(0.05)
    c = Bernoulli(probs=pheads, sample_shape=N)
    return sess.run([pheads, c])

def build_dataset(N):
    pheads = Uniform(low=0.0, high=1.0)
    c = Bernoulli(probs=pheads, sample_shape=N)
    return sess.run([pheads, c])

x = tf.range(-0.2, 1.2, 0.001)
plt.plot(*sess.run([x, Uniform(low=0.0, high=1.0).prob(x)]));
#plt.plot(*sess.run([x, Beta(concentration1=1.0, concentration0=1.0).prob(x)]));
plt.ylim((-0.2, 1.2));

# DATA
pheads_true, c_train = build_dataset(N)

pheads_true

c_train

sum(c_train == 0)

sum(c_train == 1)



pheads_fair = Beta(concentration1=1000.0, concentration0=1000.0)  # blue
pheads_unfair = Beta(concentration1=0.1, concentration0=0.1)  # green
# pheads_unknown = Uniform(low=0.0, high=1.0)  # red
pheads_unknown = Beta(concentration1=1.0, concentration0=1.0)  # red

x = tf.range(0.0, 1.0, 0.001)
plt.plot(*sess.run([x, pheads_fair.prob(x)]));
plt.plot(*sess.run([x, pheads_unfair.prob(x)]));
plt.plot(*sess.run([x, pheads_unknown.prob(x)]));
plt.axvline(x=pheads_true);

# FORWARD MODEL
pheads = pheads_unknown
c = Bernoulli(probs=pheads, sample_shape=N)

sess.run({key: val for
          key, val in six.iteritems(pheads_post.parameters)
          if isinstance(val, tf.Tensor)})

# CRITICISM
mean, stddev = sess.run([pheads_post.mean(), pheads_post.stddev()])
print("Exact posterior mean:")
print(mean)
print("Exact posterior std:")
print(stddev)

x = tf.range(0.0, 1.0, 0.001)
plt.plot(*sess.run([x, pheads.prob(x)]));  # blue
plt.plot(*sess.run([x, pheads_post.prob(x)]));  # green
plt.axvline(x=pheads_true);  # blue

# this can take a minute
fig = plt.figure()
ax = plt.axes(xlim=(-0.05, 1.05), ylim=(-1.0, 11.0))

def go(pheads_prior, sample_shape, c_train):
    # MODEL
    c = Bernoulli(probs=pheads_prior,
                  sample_shape=sample_shape)
    # INFERENCE
    pheads_cond = ed.complete_conditional(pheads_prior)
    pheads_post = ed.copy(pheads_cond, {c: c_train[:sample_shape]})
    
    # CRITICISM
    ax.plot(*sess.run([x, pheads_post.prob(x)]));
    
    # RECURSION
    if len(c_train[sample_shape:]) >= sample_shape:
        go(pheads_post, sample_shape, c_train[sample_shape:])

pheads_prior = Beta(concentration1=1.0, concentration0=1.0)
ax.plot(*sess.run([x, pheads_prior.prob(x)]));  # blue
plt.axvline(x=pheads_true);  # blue
go(pheads_prior, 33, c_train)

