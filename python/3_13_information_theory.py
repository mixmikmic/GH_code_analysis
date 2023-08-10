import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

x = [p for p in np.linspace(0.0000001,1,50)]
#fig = plt.figure()
plt.plot(x, -np.log(x))
plt.xlabel("P(event)")
plt.ylabel("Self-information")
plt.title("Self-information of an event")

def P_bern(x, lam):
    """
    Returns the probability of random variable taking on value x from a Bernoulli distribution
    whose parameter is lambda.  lambda defines the probability of observing x = 1.
    """
    assert x in [0,1]
    assert lam >= 0 and lam <= 1
    
    return lam**x * (1-lam)**(1-x)

#%%
def entropy_bern(lam):
    """
    Returns the Shannon entropy of a Bernoulli distribution with parameter lambda.
    """
    assert lam >= 0 and lam <= 1
    
    sum = 0
    # Sum over 0 and 1, the only two possible observable values in a Bernoulli trial
    for x in [0,1]:
        sum += P_bern(x, lam) * np.log(P_bern(x, lam))
    return -sum

#%%
fig = plt.figure()
lam = [p for p in np.linspace(0.0000001,1-.0000001,50)]
plt.plot(lam, [entropy_bern(a) for a in lam])
plt.xlabel("lambda")
plt.ylabel("Shannon entropy")
plt.title("Shannon entropy of Bernoulli distribution with parameter lambda")



