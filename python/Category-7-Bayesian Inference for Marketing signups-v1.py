import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Step 1: Modeling the prior - which is the belief that we have on the rate of sign-ups before we see the data
# In this case prior is assumed to be a uniform distribution between 0&1-i.e.any rate of signups between 0 & 1 are equally likely

# Number of draws from the prior
num_draws = 10000

# Each draw from the prior
sign_up_rate_prior = pd.Series(np.random.uniform(0,1,num_draws))

sign_up_rate_prior.hist()

# Step 2: Define a generative model
def generative_model(n,p):
    number_success = np.random.binomial(n,p)
    return number_success

# Step 3: Simulate the data from the prior and the generative model
number_of_emails = 60
sim_data = list()
for p in sign_up_rate_prior:
    sim_data.append(generative_model(number_of_emails,p))

# Step 4: Get the posterior distribution 
# Approximate Bayesian computation - Here you filter off all draws that do not match the data.
observed_data = 12 # Number of people actually signed up
sign_up_rate_posterior = sign_up_rate_prior[list(map(lambda x: x == observed_data, sim_data))]

sign_up_rate_posterior.hist() # Eyeball the posterior

# Summarize the posterior, where a common summary is to take the mean or the median posterior, 
# and perhaps a 95% quantile interval.


print('Number of draws left: %d, Posterior median: %.3f, Posterior quantile interval: %.3f-%.3f' % 
      (len(sign_up_rate_posterior), sign_up_rate_posterior.median(), sign_up_rate_posterior.quantile(.025), 
       sign_up_rate_posterior.quantile(.975)))

# This can be done with a for loop
number_of_people = 100
signups = list()

for p in sign_up_rate_posterior:
    signups.append(np.random.binomial(number_of_people, p))

signups = pd.Series(signups)
signups.hist()
print('Likely number of people to signup %d' %signups.median())
print('Sign-up 95%% quantile interval %d-%d' % tuple(signups.quantile([.25, .75]).values))



