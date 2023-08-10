get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from pymc3 import * 
import numpy as np 

d = np.random.normal(size=(3, 30))
d1 = d[0] + 4
d2 = d[1] + 4
yd = .2*d1 +.3*d2 + d[2]

lam = 3

with Model() as model:
    s = Exponential('s', 1)
    tau = Uniform('tau', 0, 1000)
    b = lam * tau
    m1 = Laplace('m1', 0, b)
    m2 = Laplace('m2', 0, b)
    
    p = d1*m1 + d2*m2
    
    y = Normal('y', mu=p, sd=s, observed=yd) 

with model: 
    start = find_MAP()
    
    step1 = Metropolis([m1, m2])
    
    step2 = Slice([s, tau])
    
    trace = sample(10000, [step1, step2], start=start)

traceplot(trace);

plt.figure(figsize=(10,10))
plt.hexbin(trace['m1'],trace['m2'], gridsize = 50)
plt.show()

