from __future__ import print_function, division

import numpy as np
import thinkplot
import thinkstats2

from collections import Counter

get_ipython().magic('matplotlib inline')
formats = ['png', 'pdf']

sentences = np.random.gamma(shape=2, scale=60, size=1000).astype(int)
cdf1 = thinkstats2.Cdf(sentences, label='actual')
thinkplot.PrePlot(2)
thinkplot.Cdf(cdf1)
thinkplot.Config(xlabel='sentence (months)', ylabel='CDF')
thinkplot.Save('orange1', formats=formats)

(sentences < 5*12).mean(), (sentences > 10*12).mean()

releases = sentences.cumsum()

sentences[:10]

releases[:10]

releases.searchsorted(500)

def random_sample(sentences, releases):
    i = np.random.random_integers(1, releases[-1])
    prisoner = releases.searchsorted(i)
    sentence = sentences[prisoner]
    return i, prisoner, sentence

for _ in range(10):
    print(random_sample(sentences, releases))

arrivals = np.random.random_integers(1, releases[-1], 10000)
prisoners = releases.searchsorted(arrivals)
sample = sentences[prisoners]
cdf2 = thinkstats2.Cdf(sample, label='biased')

thinkplot.PrePlot(2)
thinkplot.Cdf(cdf1)
thinkplot.Cdf(cdf2)
thinkplot.Config(xlabel='sentence (months)', ylabel='CDF', loc='lower right')
thinkplot.Save('orange2', formats=formats)

sentences.mean(), sample.mean()

def bias_pmf(pmf, t=0, label=None):
    label = label if label is not None else pmf.label
    new_pmf = pmf.Copy(label=label)

    for x, p in pmf.Items():
        new_pmf[x] *= (x + t)
        
    new_pmf.Normalize()
    return new_pmf

pmf = thinkstats2.Pmf(sentences)
biased = bias_pmf(pmf, t=0, label='BiasPmf').MakeCdf()

thinkplot.PrePlot(2)
thinkplot.Cdf(cdf2)
thinkplot.Cdf(biased)
thinkplot.Config(xlabel='sentence (months)', ylabel='CDF', loc='lower right')

def simulate_sentence(sentences, t):
    counter = Counter()
    
    releases = sentences.cumsum()
    last_release = releases[-1]
    arrival = np.random.random_integers(1, max(sentences))
    
    for i in range(arrival, last_release-t, 100):
        first_prisoner = releases.searchsorted(i)
        last_prisoner = releases.searchsorted(i+t)
    
        observed_sentences = sentences[first_prisoner:last_prisoner+1]
        counter.update(observed_sentences)
    
    print(sum(counter.values()))
    return thinkstats2.Cdf(counter, label='observed %d' % t)

def plot_cdfs(cdf3):
    thinkplot.PrePlot(2)
    thinkplot.Cdf(cdf1)
    thinkplot.Cdf(cdf2)
    thinkplot.Cdf(cdf3, color='orange')
    thinkplot.Config(xlabel='sentence (months)', ylabel='CDF', loc='lower right')

cdf13 = simulate_sentence(sentences, 13)
#cdf13 = bias_pmf(pmf, 13).MakeCdf()
plot_cdfs(cdf13)
thinkplot.Save('orange3', formats=formats)

cdf120 = simulate_sentence(sentences, 120)
#cdf120 = bias_pmf(pmf, 120).MakeCdf()
plot_cdfs(cdf120)
thinkplot.Save('orange4', formats=formats)

cdf600 = simulate_sentence(sentences, 600)
#cdf600 = bias_pmf(pmf, 600).MakeCdf()
plot_cdfs(cdf600)
thinkplot.Save('orange5', formats=formats)

thinkplot.PrePlot(cols=3)
plot_cdfs(cdf13)
thinkplot.SubPlot(2)
plot_cdfs(cdf120)
thinkplot.SubPlot(3)
plot_cdfs(cdf600)
thinkplot.Save('orange6', formats=formats)



