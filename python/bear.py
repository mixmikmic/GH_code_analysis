from __future__ import print_function, division

import thinkbayes2
import thinkplot

import numpy as np
from scipy import stats

get_ipython().magic('matplotlib inline')

data = {
    2008: ['Gardiner', 'McNatt', 'Terry'],
    2009: ['McNatt', 'Ryan', 'Partridge', 'Turner', 'Demers'],
    2010: ['Gardiner', 'Barrett', 'Partridge'],
    2011: ['Barrett', 'Partridge'],
    2012: ['Sagar'],
    2013: ['Hammer', 'Wang', 'Hahn'],
    2014: ['Partridge', 'Hughes', 'Smith'],
    2015: ['Barrett', 'Sagar', 'Fernandez'],
}

def MakeBinomialPmf(n, p):
    ks = range(n+1)
    ps = stats.binom.pmf(ks, n, p)
    pmf = thinkbayes2.Pmf(dict(zip(ks, ps)))
    pmf.Normalize()
    return pmf

class Bear1(thinkbayes2.Suite, thinkbayes2.Joint):
    def Likelihood(self, data, hypo):
        n, p = hypo
        like = 1
        for year, sobs in data.items():
            k = len(sobs)
            if k > n:
                return 0
            like *= stats.binom.pmf(k, n, p)
        return like
    
    def Predict(self):
        metapmf = thinkbayes2.Pmf()
        for (n, p), prob in bear.Items():
            pmf = MakeBinomialPmf(n, p)
            metapmf[pmf] = prob
        mix = thinkbayes2.MakeMixture(metapmf)
        return mix

hypos = [(n, p) for n in range(15, 70) 
                for p in np.linspace(0, 1, 101)]
bear = Bear1(hypos)

bear.Update(data)

pmf_n = bear.Marginal(0)
thinkplot.PrePlot(5)
thinkplot.Pdf(pmf_n, label='n')
thinkplot.Config(xlabel='Number of runners (n)', 
                 ylabel='PMF', loc='upper right')
pmf_n.Mean()

pmf_p = bear.Marginal(1)
thinkplot.Pdf(pmf_p, label='p')
thinkplot.Config(xlabel='Probability of showing up (p)', 
                 ylabel='PMF', loc='upper right')
pmf_p.CredibleInterval(95)

thinkplot.Contour(bear, pcolor=True, contour=False)
thinkplot.Config(xlabel='Number of runners (n)',
                 ylabel='Probability of showing up (p)',
                 ylim=[0, 0.4])

predict = bear.Predict()
thinkplot.Hist(predict, label='k')
thinkplot.Config(xlabel='# Runners who beat me (k)',
                 ylabel='PMF', xlim=[-0.5, 12])
predict[0]

ss = thinkbayes2.Beta(2, 1)
thinkplot.Pdf(ss.MakePmf(), label='S')
thinkplot.Config(xlabel='Probability of showing up (S)',
                 ylabel='PMF', loc='upper left')

os = thinkbayes2.Beta(3, 1)
thinkplot.Pdf(os.MakePmf(), label='O')
thinkplot.Config(xlabel='Probability of outrunning me (O)',
                 ylabel='PMF', loc='upper left')

bs = thinkbayes2.Beta(1, 1)
thinkplot.Pdf(bs.MakePmf(), label='B')
thinkplot.Config(xlabel='Probability of being in my age group (B)',
                 ylabel='PMF', loc='upper left')

n = 1000
sample = ss.Sample(n) * os.Sample(n) * bs.Sample(n)
cdf = thinkbayes2.Cdf(sample)

thinkplot.PrePlot(1)

prior = thinkbayes2.Beta(1, 3)
thinkplot.Cdf(prior.MakeCdf(), color='grey', label='Model')
thinkplot.Cdf(cdf, label='SOB sample')
thinkplot.Config(xlabel='Probability of displacing me',
                 ylabel='CDF', loc='lower right')

from itertools import chain
from collections import Counter

counter = Counter(chain(*data.values()))
len(counter), counter

def MakeBeta(count, num_races, precount=3):
    beta = thinkbayes2.Beta(1, precount)
    beta.Update((count, num_races-count))
    return beta

num_races = len(data)
betas = [MakeBeta(count, num_races) 
         for count in counter.values()]

[beta.Mean() for beta in betas]

class Bear2(thinkbayes2.Suite, thinkbayes2.Joint):

    def ComputePmfs(self, data):
        num_races = len(data)
        counter = Counter(chain(*data.values()))
        betas = [MakeBeta(count, num_races) 
                 for count in counter.values()]
        
        self.pmfs = dict()
        low = len(betas)
        high = max(self.Values())
        for n in range(low, high+1):
            self.pmfs[n] = self.ComputePmf(betas, n, num_races)
    
    def ComputePmf(self, betas, n, num_races, label=''):
        no_show = MakeBeta(0, num_races)
        all_betas = betas + [no_show] * (n - len(betas))
        
        ks = []
        for i in range(2000):
            ps = [beta.Random() for beta in all_betas]
            xs = np.random.random(len(ps))
            k = sum(xs < ps)
            ks.append(k)
            
        return thinkbayes2.Pmf(ks, label=label)
    
    def Likelihood(self, data, hypo):
        n = hypo
        k = data
        return self.pmfs[n][k]
    
    def Predict(self):
        metapmf = thinkbayes2.Pmf()
        for n, prob in self.Items():
            pmf = bear2.pmfs[n]
            metapmf[pmf] = prob
        mix = thinkbayes2.MakeMixture(metapmf)
        return mix

bear2 = Bear2()

thinkplot.PrePlot(3)
pmf = bear2.ComputePmf(betas, 18, num_races, label='n=18')
pmf2 = bear2.ComputePmf(betas, 22, num_races, label='n=22')
pmf3 = bear2.ComputePmf(betas, 26, num_races, label='n=24')
thinkplot.Pdfs([pmf, pmf2, pmf3])
thinkplot.Config(xlabel='# Runners who beat me (k)',
                 ylabel='PMF', loc='upper right')

low = 15
high = 35
bear2 = Bear2(range(low, high))
bear2.ComputePmfs(data)

for year, sobs in data.items():
    k = len(sobs)
    bear2.Update(k)

thinkplot.PrePlot(1)
thinkplot.Pdf(bear2, label='n')
thinkplot.Config(xlabel='Number of SOBs (n)',
                 ylabel='PMF', loc='upper right')

predict = bear2.Predict()

thinkplot.Hist(predict, label='k')
thinkplot.Config(xlabel='# Runners who beat me (k)', ylabel='PMF', xlim=[-0.5, 12])
predict[0]





