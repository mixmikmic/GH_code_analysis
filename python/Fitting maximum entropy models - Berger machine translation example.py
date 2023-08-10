import maxentropy.skmaxent

a_grave = u'\u00e0'

samplespace = ['dans', 'en', a_grave, 'au cours de', 'pendant']


def f0(x):
    return x in samplespace

def f1(x):
    return x=='dans' or x=='en'

def f2(x):
    return x=='dans' or x==a_grave

f = [f0, f1, f2]

model = maxentropy.skmaxent.MinDivergenceModel(f, samplespace, vectorized=False, verbose=True)

# Now set the desired feature expectations
K = [1.0, 0.3, 0.5]

import numpy as np

np.array(K, ndmin=2)

# Fit the model
model.fit(np.array(K, ndmin=2))

model.F.todense()

# Output the distribution
print("\nFitted model parameters are:\n" + str(model.params))

print("\nFitted distribution is:")
p = model.probdist()
for j in range(len(model.samplespace)):
    x = model.samplespace[j]
    print("\tx = %-15s" %(x + ":",) + " p(x) = "+str(p[j]))

# Now show how well the constraints are satisfied:
print()
print("Desired constraints:")
print("\tp['dans'] + p['en'] = 0.3")
print("\tp['dans'] + p['à']  = 0.5")
print()
print("Actual expectations under the fitted model:")
print("\tp['dans'] + p['en'] =", p[0] + p[1])
print("\tp['dans'] + p['à']  = " + str(p[0]+p[2]))
# (Or substitute "x.encode('latin-1')" if you have a primitive terminal.)

import numpy as np
np.allclose(model.expectations(), K)

