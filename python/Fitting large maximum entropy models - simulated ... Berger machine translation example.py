from __future__ import print_function

import sys

import maxentropy
from maxentropy.maxentutils import dictsampler

import numpy as np

samplespace = ['dans', 'en', 'à', 'au cours de', 'pendant']

@np.vectorize
def f0(x):
    return x in samplespace

@np.vectorize
def f1(x):
    return x == 'dans' or x == 'en'

@np.vectorize
def f2(x):
    return x == 'dans' or x == 'à'

f = [f0, f1, f2]

f0('dans')

# Define a uniform instrumental distribution for sampling
samplefreq = {e: 1 for e in samplespace}

auxiliary_sampler = dictsampler(samplefreq, size=10**5, return_probs='logprob')

next(auxiliary_sampler)

model = maxentropy.BigModel(auxiliary_sampler)

# Default: model.algorithm = 'CG'
# Can choose from ['CG', 'BFGS', 'LBFGSB', 'Powell', 'Nelder-Mead']

# Now set the desired feature expectations
K = [1.0, 0.3, 0.5]

from maxentropy.maxentutils import importance_sampler, create_vectorized_feature_function

features = create_vectorized_feature_function(f, sparse=False)

xs, logprobs = next(auxiliary_sampler)

xs

features(xs)

model.samplegen = importance_sampler(features, auxiliary_sampler)
model.reset(len(f))

next(model.samplegen)

model.resample()

model.verbose = True

# Fit the model
# model.avegtol = 1e-5
model.fit(f, K)

# Output the true distribution
print("Fitted model parameters are:")
model.params

smallmodel = maxentropy.Model(samplespace)
smallmodel.setparams(model.params)

smallmodel.params

smallmodel.setfeatures(f)

smallmodel.F.todense()

F = smallmodel.F.todense().T
F

smallmodel.params

F.dot(smallmodel.params)

smallmodel.F.T.dot(smallmodel.params)

print("\nFitted distribution is:")
smallmodel.showdist()

# Now show how well the constraints are satisfied:
print()
print("Desired constraints:")
print("\tp['dans'] + p['en'] = 0.3")
print("\tp['dans'] + p['à']  = 0.5")
print()
print("Actual expectations under the fitted model:")
print("\tp['dans'] + p['en'] =", p[0] + p[1])
print("\tp['dans'] + p['à']  = " + str(p[0]+p[2]))

print("\nEstimated error in constraint satisfaction (should be close to 0):\n"
        + str(abs(model.expectations() - K)))
print("\nTrue error in constraint satisfaction (should be close to 0):\n" +
        str(abs(smallmodel.expectations() - K)))



