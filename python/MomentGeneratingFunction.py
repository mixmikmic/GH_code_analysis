get_ipython().magic('pylab inline')

import sympy as S
from sympy import stats
p = S.symbols('p',positive=True,real=True)
x=stats.Binomial('x',5,p)
t = S.symbols('t',real=True,positive=True)
mgf = stats.E(S.exp(t*x))

from IPython.display import Math

Math(S.latex(S.simplify(mgf)))

import scipy.stats

def gen_sample(ns=100,n=10):
    out =[]
    for i in range(ns):
        p=scipy.stats.uniform(0,1).rvs()
        out.append(scipy.stats.binom(n,p).rvs())
    return out

from collections import Counter
xs = gen_sample(1000)
hist(xs,range=(1,10),align='mid')
Counter(xs)

S.var('x:2',real=True)
S.var('mu:2',real=True)
S.var('sigma:2',positive=True)
S.var('t',positive=True)
x0=stats.Normal(x0,mu0,sigma0)
x1=stats.Normal(x1,mu1,sigma1)

mgf0=S.simplify(stats.E(S.exp(t*x0)))
mgf1=S.simplify(stats.E(S.exp(t*x1)))
mgfY=S.simplify(mgf0*mgf1)

Math(S.latex(mgf0))

Math(S.latex(S.powsimp(mgfY)))

S.collect(S.expand(S.log(mgfY)),t)



