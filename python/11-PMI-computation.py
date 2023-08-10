import collections
from itertools import combinations
import pickle

def flatten(it):
    for x in it:
        if (isinstance(x, collections.Iterable) and
            not isinstance(x, tuple) and
            not isinstance(x, str)):
            yield from flatten(x)
        else:
            yield x

streaks = pickle.load(open('../data/streaks.pickle', 'rb'))

streaks[0:1]

for strk in streaks:
    for i, item in enumerate(strk):
        strk[i] = item.loc['group'].lower()

streaks

joint_counts = collections.Counter(flatten((p for p in combinations(strk, 2)) for strk in streaks))
prior_counts = collections.Counter(flatten((x for x in strk) for strk in streaks))

all_pairs_total_count = sum(joint_counts.values())
orders_total_count = sum(prior_counts.values())

joint_probs = {}
prior_probs = {}

for pair, count in joint_counts.items():
    joint_probs[pair] = count / all_pairs_total_count
    
for item, count in prior_counts.items():
    prior_probs[item] = count / orders_total_count

print(sum(joint_probs.values()))
print(sum(prior_probs.values()))

from math import log
pmi = collections.Counter({(x,y): log(joint_probs[(x,y)] / (prior_probs[x]*prior_probs[y])) for x,y in joint_probs.keys()})

[x[0] for x in pmi.most_common(20)]

max([joint_counts[x[0]] for x in pmi.most_common(20000)])

with open('group_pmi.pickle', 'wb') as f:
    pickle.dump(pmi, f)

len(pmi)

import matplotlib.pyplot as plt

sjp = sorted(joint_probs.values(),reverse=True)
spp = sorted(prior_probs.values(), reverse = True)
spmi = sorted(pmi.values(), reverse = True)



num = 5000
plt.plot(range(num), spmi[-num:])
plt.show()

plt.plot(range(num), spp[:num])
plt.show()



