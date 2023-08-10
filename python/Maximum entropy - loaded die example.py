get_ipython().magic('matplotlib inline')
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import maxentropy

samplespace = np.arange(6) + 1

def f0(x):
    return x

f = [f0]



model = maxentropy.Model(samplespace)
model.verbose = True

model.avegtol

# Now set the desired feature expectations
K = [4.5]

# Fit the model
model.fit(f, K)

model.params

# Output the distribution
print("\nFitted model parameters are:\n" + str(model.params))
print("The fitted distribution is:")
model.showdist()

# Now show how well the constraints are satisfied:
print()
print("Desired constraints:")
print("\tE(X) = 4.5")
print()
print("Actual expectations under the fitted model:")
print("\t\hat{X} = ", model.expectations())

np.allclose(K, model.expectations())

model.algorithm = 'BFGS'
model.verbose = False
model.fit(f, K)

np.allclose(K, model.expectations())

model.params

model.probdist()

sns.barplot(np.arange(1, 7), model.probdist())
plt.title('Probability $p(x)$ of each die face $x$')

