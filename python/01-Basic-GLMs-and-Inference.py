import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

grad = pd.read_csv("https://stats.idre.ucla.edu/stat/data/binary.csv")

# Let's do some EDA!
print(grad.shape)
grad.head()

grad.describe()

grad.admit.value_counts()

grad.boxplot('gre', by = 'admit')
grad.boxplot(['gpa', 'rank'], by = 'admit')

# Now, let's build our GLM
indep_vars = ['gre', 'gpa', 'rank']
x_mat = sm.add_constant(grad[indep_vars])
y_vec = grad.admit
glm_logit = sm.GLM(y_vec, 
                   x_mat,
                   sm.families.Binomial(sm.families.links.logit)).fit()

glm_logit.summary()

# Next, let's see if we can safely reduce our model
reduced_vars = ['gre', 'gpa']
x_reduced = sm.add_constant(grad[reduced_vars])
glm_reduced = sm.GLM(y_vec,
                    x_reduced,
                    sm.families.Binomial(sm.families.links.logit)).fit()
glm_reduced.summary()

# Test model differences
from scipy.stats import chi2

D = glm_reduced.deviance - glm_logit.deviance
print('Difference in Deviance: ', D)
pval = 1 - chi2.cdf(D, df = 1)
print('p-value of test of difference: ', pval) # What can we conclude here?

award = pd.read_csv("https://stats.idre.ucla.edu/stat/data/poisson_sim.csv")

# Let's do some EDA:
award.head()

award.plot('math', 'num_awards', kind = 'scatter')
award.boxplot('num_awards', by = 'prog') # Oh no - this is hideous.  Why can't everything be ggplot?

# Notice that prog is actually a categorical variable - I am aware of this.
# I'm going to suspsend that knowledge for the sake of example.
poi_vars = ['prog', 'math']
x_poi = sm.add_constant(award[poi_vars])
y_poi = award.num_awards

glm_poi = sm.GLM(y_poi,
                 x_poi,
                 family = sm.families.Poisson()).fit()
glm_poi.summary()

# Interesting - it looks like prog is pretty insignificant.
# Let's try removing it and seeing what happens.
x_poi_red = sm.add_constant(award.math)

glm_poi_red = sm.GLM(y_poi,
                     x_poi_red,
                     family = sm.families.Poisson()).fit()
glm_poi_red.summary()

# Test model difference:
D_poi = glm_poi_red.deviance - glm_poi.deviance
print('Difference in Deviance: ', D_poi)
pval_poi = 1 - chi2.cdf(D_poi, 1)
print('p-value of test of difference: ', pval_poi) # What does this mean?

