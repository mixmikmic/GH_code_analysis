get_ipython().magic('matplotlib qt4')
from __future__ import division

from models import tools, filters

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import defaultdict

sns.set_style("ticks", {"legend.frameon": True})
mpl.rcParams['axes.color_cycle'] = ['#02A5F4', 'orange', 'green']

data = tools.load_data(limit=3000000, offset=100000)
data = data[data['response_time'] < 20000]

data = data[filters.countries(data)]

_, bins = pd.cut(data['response_time'], bins=20, retbins=True)
intervals = zip(bins[:-1], bins[1:])

responses = defaultdict(lambda: [])
for lower_bound, upper_bound in intervals:
    tools.echo('{}-{}'.format(lower_bound, upper_bound))
    for place in data['place_id'].unique():
        vals = data[(data['response_time'] >= lower_bound) &
                    (data['response_time'] < upper_bound) &
                    (data['place_id'] == place)]
        responses[place].append(vals['is_correct'].mean())

X = [[] for _ in intervals]
for place in responses:
    for i, value in enumerate(responses[place]):
        if np.isfinite(value):
            X[i].append(value)

labels = ['({}, {}]'.format(int(i), int(j)) for i, j in intervals]

plt.figure(num=None, figsize=(9, 6), dpi=120)
plt.xticks(rotation=70)
bp = plt.boxplot(X, labels=labels, showfliers=False)
plt.xlabel('Response time in miliseconds')
plt.ylabel('Probability of recall')
plt.subplots_adjust(bottom=0.25)

plt.setp(bp['medians'], color='orange')
plt.setp(bp['boxes'], color='#02A5F4')
plt.setp(bp['whiskers'], color='#02A5F4')
plt.setp(bp['fliers'], color='#02A5F4', marker='+')

plt.tight_layout()

previous_is_correct = {}
groups = data.groupby(['user_id', 'place_id'])

for i, (_, group) in enumerate(groups):
    prev_idx = None
    for idx in sorted(group.index):
        if prev_idx is not None:
            previous_is_correct[idx] = group.ix[prev_idx]['is_correct']
        prev_idx = idx
    if i % 10000 == 0:
        tools.echo(i)

d1 = data
d1['response_bin'] = d1['response_time'] // 500
d1 = d1[['is_correct', 'response_bin']]

d2 = pd.DataFrame(previous_is_correct.items(), columns=['id', 'previous_correct'])
d2 = d2.set_index('id')

d = pd.concat([d1, d2], axis=1, join='inner')

prev_incorrect = d[d['previous_correct'] == 0]
prev_correct = d[d['previous_correct'] == 1]

def grouping(df):
    gs = df[['is_correct', 'response_bin']].groupby(['response_bin'])
    return gs.sum() / gs.count()

plt.figure(num=None, figsize=(5, 4), dpi=120)
plt.plot(grouping(prev_correct), '.-', label='previous correct')
plt.plot(grouping(prev_incorrect), '.-', label='previous incorrect')
plt.xlabel('Response time in seconds')
plt.ylabel('Success')
legend = plt.legend(loc='lower right', prop={'size': 12})
legend.get_frame().set_linewidth(1)
plt.xticks(range(0, 21, 2))
plt.tight_layout()

plt.figure(num=None, figsize=(5, 4), dpi=120)
plt.hist([list(prev_correct['response_bin']),
          list(prev_incorrect['response_bin'])],
         bins=20, rwidth=0.8,
         label=['previous correct', 'previous incorrect'])
plt.yscale('log')
plt.xlabel('Response time in seconds')
plt.ylabel('Number of answers')
plt.xticks(range(0, 21, 2))
legend = plt.legend(prop={'size': 12})
legend.get_frame().set_linewidth(1)
plt.tight_layout()

print 'Previous correct:'
print grouping(prev_correct).to_dict()
print ''
print 'Previous incorrect:'
print grouping(prev_incorrect).to_dict()



