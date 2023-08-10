from __future__ import print_function, division

get_ipython().magic('matplotlib inline')
get_ipython().magic('precision 3')

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from thinkbayes2 import Suite, Beta
from thinkbayes2 import MakeMixture

import thinkplot

def make_beta(hits, at_bats, alpha=1, beta=1, label=None):
    beta = Beta(alpha, beta, label)
    beta.Update((hits, at_bats-hits))
    return beta

prior = make_beta(0, 0, 1, 1).MakePmf(steps=1001, label='prior')
print(prior.Mean())
thinkplot.Pdf(prior)

def compare(beta1, beta2, xlabel, steps=1001):
    pmf1 = beta1.MakePmf(steps=steps)
    pmf2 = beta2.MakePmf(steps=steps)
    print(pmf1.MAP(), pmf2.MAP())
    print('%3.3f %3.3f' % (pmf1.Mean(), pmf2.Mean()))
    print('%3.3f' % (pmf1 > pmf2))
    thinkplot.Pdfs([pmf1, pmf2])
    thinkplot.Config(xlabel=xlabel,
                    ylabel='PMF')

xlabel = 'batting average'
jeter95 = make_beta(12, 48, label='jeter95')
justice95 = make_beta(104, 411, label='justice95')
compare(jeter95, justice95, xlabel)

jeter96 = make_beta(183, 582, label='jeter96')
justice96 = make_beta(45, 140, label='justice96')
compare(jeter96, justice96, xlabel)

jeter_combined = make_beta(195, 630, label='jeter_combined')
justice_combined = make_beta(149, 551, label='justice_combined')
compare(jeter_combined, justice_combined, xlabel)

prior = make_beta(0, 0, 8, 25).MakePmf(steps=1001, label='prior')
print(prior.Mean())
thinkplot.Pdf(prior)

alphas = 8, 25
jeter95 = make_beta(12, 48, *alphas, label='jeter95')
justice95 = make_beta(104, 411, *alphas, label='justice95')
compare(jeter95, justice95, xlabel)

jeter96 = make_beta(183, 582, *alphas, label='jeter96')
justice96 = make_beta(45, 140, *alphas, label='justice96')
compare(jeter96, justice96, xlabel)

jeter_combined = make_beta(195, 630, *alphas, label='jeter_combined')
justice_combined = make_beta(149, 551, *alphas, label='justice_combined')
compare(jeter_combined, justice_combined, xlabel)

alphas = 1, 1
xlabel = 'Prob success'
Asmall = make_beta(81, 87, *alphas, label='Asmall')
Bsmall = make_beta(234, 270, *alphas, label='Bsmall')
compare(Asmall, Bsmall, xlabel)
thinkplot.config(loc='upper left')

Alarge = make_beta(192, 263, *alphas, label='Alarge')
Blarge = make_beta(55, 80, *alphas, label='Blarge')
compare(Alarge, Blarge, xlabel)

Acombo = make_beta(273, 350, *alphas, label='Acombo')
Bcombo = make_beta(289, 350, *alphas, label='Bcombo')
compare(Acombo, Bcombo, xlabel)

prior = make_beta(0, 0, 10, 2).MakePmf(steps=1001, label='prior')
print(prior.Mean())
thinkplot.Pdf(prior)

alphas = 10, 2
xlabel = 'Prob success'
Asmall = make_beta(81, 87, *alphas, label='Asmall')
Bsmall = make_beta(234, 270, *alphas, label='Bsmall')
compare(Asmall, Bsmall, xlabel)
thinkplot.config(loc='upper left')

Alarge = make_beta(192, 263, *alphas, label='Alarge')
Blarge = make_beta(55, 80, *alphas, label='Blarge')
compare(Alarge, Blarge, xlabel)

Acombo = make_beta(273, 350, *alphas, label='Acombo')
Bcombo = make_beta(289, 350, *alphas, label='Bcombo')
compare(Acombo, Bcombo, xlabel)

class Kidney(Suite):
    def __init__(self, alpha, beta, label=None):
        pmf = Beta(alpha, beta, label).MakePmf(steps=1001)
        Suite.__init__(self, pmf)
        
    def Likelihood(self, data, hypo):
        hits, at_bats = data
        misses = at_bats - hits
        p = hypo
        like = p**hits * (1-p)**misses
        return like

alphas = 10, 2

def make_suite(data, label=None):
    suite = Kidney(*alphas, label=label)
    like = suite.Update(data)
    return like, suite

like1, Asmall = make_suite((81, 87), 'Asmall')
like2, Bsmall = make_suite((234, 270), 'Bsmall')
print(like1, like2)

thinkplot.Pdfs([Asmall, Bsmall])

like1, Acombo = make_suite((273, 350), 'Acombo')
like2, Bcombo = make_suite((289, 350), 'Bcombo')
print(like1, like2)

thinkplot.Pdfs([Acombo, Bcombo])

import pandas as pd

iterables = [['A', 'B'], ['S', 'L'], ['pos', 'neg']]
index = pd.MultiIndex.from_product(iterables, names=['treatment', 'size', 'outcome'])



data = pd.Series([81, 6, 192, 71, 234, 36, 55, 25], index=index)
data.sort_index(inplace=True)

data

data['A']

data.loc['A', 'S']

data.loc['A'].sum(axis=0) 

data.loc['B'].sum(axis=0) 

X = slice(None)
data.loc['A', X, 'pos'].sum()

indices = ['A', 'B']
for index in indices:
    print(data[index])

from itertools import product

for index in product('AB'):
    print(data[index])

indices = ['A', 'B']
for index in product('AB', 'SL'):
    print(data[index])

iterables = [['A', 'B'], ['S', 'L']]
index = pd.MultiIndex.from_product(iterables, names=['treatment', 'size'])

data = np.array([81, 6, 192, 71, 234, 36, 55, 25]).reshape((4, 2))
data

df = pd.DataFrame(data, index=index, columns=['pos', 'neg'])
df.sort_index(inplace=True)
df

df.loc['A']

for index in product('AB'):
    print(index)
    print(df.loc[index])

for index in product('AB', 'SL'):
    print(index)
    group = df.loc[index]
    print(group.pos, group.neg)

from scipy import stats

class Model:
    def __init__(self, pos, neg, alphas=(0, 0)):
        self.pos = pos
        self.neg = neg
        self.beta = Beta(*alphas)
        self.beta.Update((pos, neg))
        self.pmf = self.beta.MakePmf(steps=101)
        del self.pmf[0]
        del self.pmf[1]
        
    def mean(self):
        return self.beta.Mean()
        
    def MAP(self):
        return self.beta.MAP()
        
    def deviance(self, p):
        k = self.pos
        n = self.pos + self.neg
        logprob = stats.binom.logpmf(k, n, p)
        return -2 * logprob
    
    def deviance_hat(self):
        return self.deviance(self.MAP())
    
    def deviance_avg(self):
        return self.pmf.Expect(self.deviance)
    
    def DIC(self):
        return 2 * self.deviance_hat() - self.deviance_avg()

def print_models(models):
    total = 0
    for index, model in sorted(models.items()):
        dic = model.DIC()
        total += dic
        print(index, model.deviance_hat(), model.deviance_avg(), dic)
    return total

models = {}
for index in product('AB', 'S'):
    group = df.loc[index]
    model = Model(group.pos, group.neg)
    models[index] = model
    
print_models(models)

models = {}
for index in product('AB', 'L'):
    group = df.loc[index]
    model = Model(group.pos, group.neg)
    models[index] = model
    
print_models(models)

models = {}
for index in product('AB'):
    group = df.loc[index].sum()
    model = Model(group.pos, group.neg)
    models[index] = model

print_models(models)

# Encapsulate the model-building loop
# Run the same analysis on the batting averages



