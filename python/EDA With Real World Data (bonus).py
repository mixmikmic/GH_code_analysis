get_ipython().magic('matplotlib inline')

import json
import operator

from collections import namedtuple, Counter, OrderedDict

import matplotlib as mpl
import matplotlib.pyplot as plt

from IPython.display import set_matplotlib_formats

set_matplotlib_formats('retina')

with open('./data/sfpd.json') as fin:
    data = json.load(fin)
columns = [obj['name'] for obj in data['meta']['view']['columns']]
Incident = namedtuple('Incident', columns)
incidents = [Incident(*row) for row in data['data']]    

def dhist(d):
    """Plots a dict object in a bar graph to match the histogram layout.
    """
    # For a bar graph, the first argument is the values at which the 
    # bars will be place. We set the width to 1 (default is 0.8) to 
    # remove any whitespace between bars and make it look more like the 
    # histogram. We set the align to 'edge' (default is 'center') to make
    # sure that the left edge of the bar is aligned with its value.
    plt.bar(range(len(d)), d.values(), width=1, align='edge')
    # Since the bars' left edges are aligned with their values, we have to 
    # move the x-tick labels in a bit to make them centered within the bar.
    # Since the bars are all a width of 1, we simply need to add 0.5 to the
    # left values of every tick. Finally, we rotate the labels 90 degrees 
    # so they are readable.
    plt.xticks([i + 0.5 for i in range(len(d))], d.keys(), rotation=90)
    # Making the graph as tight as possible by limiting the X-axis values.
    plt.xlim([0, len(d)]);

hist = Counter(i.PdDistrict for i in incidents)
hist = OrderedDict(sorted(hist.items(), key=operator.itemgetter(0)))
dhist(hist)

set(i.Category for i in incidents)

violent_crimes = ['ARSON', 'ASSAULT','KIDNAPPING', 'SEX OFFENSES, FORCIBLE']
assaults = [i for i in incidents if i.Category in violent_crimes]
hist = Counter(i.PdDistrict for i in assaults)
hist = OrderedDict(sorted(hist.items(), key=operator.itemgetter(0)))
dhist(hist)

hist = Counter(i.Category for i in incidents)
hist = OrderedDict(sorted(hist.items(), key=operator.itemgetter(1)))
plt.figure(figsize=(20, 8))
dhist(hist)

violent_crimes = ['ARSON', 'ASSAULT','KIDNAPPING', 'SEX OFFENSES, FORCIBLE']
hist = Counter(int(i.Time[:2]) for i in incidents if i.Category in violent_crimes)
morning = range(6, 12)
daytime = [12] + range(1, 13)
late_night = range(1, 6)
keys = ['%s AM' % h for h in morning] +        ['%s PM' % h for h in daytime] +        ['%s AM' % h for h in late_night]
values = [hist[k] for k in range(6, 24) + range(0, 6)]
hist = OrderedDict(zip(keys, values))
plt.figure(figsize=(12, 6))
dhist(hist)

keys = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

fig = plt.figure(figsize=(8, 2))

fig.add_subplot(1, 2, 1)
# Adjust the y limits for the current axes.
plt.ylim([14000, 20000])
hist = Counter(i.DayOfWeek for i in incidents)
values = [hist[k] for k in keys]
dhist(OrderedDict(zip(keys, values)))

fig.add_subplot(1, 2, 2)
plt.ylim([1400, 2000])
hist = Counter(i.DayOfWeek for i in incidents if i.Category in violent_crimes)
values = [hist[k] for k in keys]
dhist(OrderedDict(zip(keys, values)))

