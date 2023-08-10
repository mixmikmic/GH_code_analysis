get_ipython().magic('pylab inline')
import pylab
import seaborn as sns

import numpy as np

M = 150    # maximum age
N = 10000   # number of independent samples
K = 200    # resolution of PDFs

# Start by computing the hidden (unobserved) parameters
# using some hyperparameters alpha = (mu, std).
# Neither of these will be accessible because that is what we're
# trying to figure out! However the knowledge of how the hyperparameters
# generated the hidden parameters will become a part of the model.
mu_ANSWER = 75
std_ANSWER = 16
Z_ANSWER = std_ANSWER * np.random.randn(N) + mu_ANSWER

# From these, draw the input (observed) parameters x_i ~ U[0, z_i]
# which WILL be accessible to the code / etc. 
# "we are equally likely to meet a man at any point in his life"
# and the model will again make use of this knowledge.
X = np.asarray([np.random.uniform(high=th) for th in Z_ANSWER])
assert (0 <= X).all() and (X <= M).all()

# So far we have generated the sample data in the forward direction
# (alpha -> Z -> X). Now the inference problem is essentially the reverse
# direction (X -> (Z, alpha)) using knowledge of these distributions.

pylab.figure()
pylab.title("Sampled Data")
sns.kdeplot(X, label=r"Observed Ages $X$ (Input)")
sns.kdeplot(Z_ANSWER, label=r"Unobserved Lifespans $Z$ (Hidden)")
sns.kdeplot(Z_ANSWER - X, label=r"Remaining Life $Z - X$", linestyle='--')
pylab.xlim(0, M)
pylab.xlabel("Age")
pylab.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
pylab.show()

# Note that X has the same distribution as Z - X.
# This is not a coincidence. It has to do with way in which X was generated;
# in particular z_i - x_i ~ U[0, z_i].

# Now we want to recover Z (and alpha) from X.
# But even though X has the same distribution as Z - X
# that does NOT mean X = Z - X <=> Z = 2*X
# Equality of distributions does not imply equality of samples!

pylab.figure()
pylab.title("An Incorrect Naive Approach")
sns.kdeplot(2*X, label=r"$2X$")
sns.kdeplot(Z_ANSWER, label=r"Unobserved Lifespans $Z$ (Hidden)")
pylab.xlim(0, M)
pylab.xlabel("Age")
pylab.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
pylab.show()

# The fact that the distributions are the same, however, does give
# hope that it should be possible to recover the distribution of the
# (unobserved) hidden variables.

axis = np.linspace(0, M, K)

def normal(mu, std):
    return np.exp(-(axis - mu)**2 / (2*std**2)) / np.sqrt(2*std**2*np.pi)

def uniform_posterior(x_i):
    l = np.zeros_like(axis)
    l[axis > x_i] = 1. / axis[axis > x_i]
    return l

likelihood = np.empty((N, K))
for i, x_i in enumerate(X):
    likelihood[i, :] = uniform_posterior(x_i)
    
print("Likelihood has %d million elements" % (likelihood.size // 1000000))

mu = mu_ANSWER    # <-- cheating!
std = std_ANSWER  # <-- cheating!
prior = normal(mu, std)

Z_median = np.empty_like(X)
#Z_map = np.empty_like(X)
for i in range(N):
    joint = likelihood[i] * prior  # proportional to posterior
    cpdf = np.cumsum(joint)
    Z_median[i] = axis[np.where(cpdf >= (cpdf[-1] / 2))[0][0]]  # Median estimate
    #Z_map[i] = axis[np.argmax(joint)]  # MAP estimate

# Here we graph the difference in the true Z versus the estimate Z
# for the median estimator. And then we can compare this to using
# 2*X as a naive estimator, which is worse. We can also always guess
# mu, which is almost as good (surprisingly, since it does not even 
# depend on X). But the optimal estimate is still more sharply peaked.

def measure(r):
    return r"($%d \pm %d$)" % (np.mean(r), np.std(r))

pylab.figure()
pylab.title("Estimate Errors")
sns.kdeplot(Z_median - Z_ANSWER, label=r"Median %s" % measure(Z_median - Z_ANSWER))
#sns.kdeplot(Z_map - Z_ANSWER, label=r"MAP %s" % measure(Z_map - Z_ANSWER))
sns.kdeplot(mu - Z_ANSWER, label=r"$\mu$ %s" % measure(mu - Z_ANSWER))
sns.kdeplot(2*X - Z_ANSWER, label=r"$2X$ %s" % measure(2*X - Z_ANSWER))
pylab.xlabel(r"$\hat{Z} - Z$")
pylab.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
pylab.show()

max_L = float('-inf')
best_alpha = None

for mu in np.linspace(70, 80, 5):
    for std in np.linspace(10, 22, 3):
        prior = normal(mu, std)
 
        L = 0  # log-likelihood
        for i in range(N):
            L += np.log(np.dot(likelihood[i], prior))
        
        if L > max_L:
            max_L = L
            best_alpha = (mu, std)
            
        print(mu, std, L)
        
print("Optimal hyperparameters:", best_alpha)
print("Actual hyperparameters:", (mu_ANSWER, std_ANSWER))

# Start with an initial guess for the hyperparameters
mu = 90
std = 5
num_steps = 100
alpha = np.empty((num_steps, 2))

# And then use estimates of Z to estimate the new prior
from nengo.utils.progress import ProgressTracker
with ProgressTracker(1, True, "Initializing"): pass  # Hack to fix bug
with ProgressTracker(2 * num_steps, True, "Expectation-Maximization") as progress:
    for k in range(num_steps):
        prior = normal(mu, std)

        def sum_over(g):
            #s = 0
            #for i in range(N):
            #    s += np.dot(g, likelihood[i] * prior) / np.dot(likelihood[i], prior)
            return np.sum(  # numpy optimized version of above comment
                np.dot(g, likelihood.T * prior[:, None]) /
                np.dot(likelihood, prior))

        mu = sum_over(axis) / N
        progress.step()
        std = np.sqrt(sum_over((axis - mu)**2) / N)
        progress.step()

        alpha[k, :] = [mu, std]

print("Final hyperparameters:", mu, std)

cpal = sns.color_palette(None, 2)

pylab.figure()
pylab.title("Expectation-Maximization")
pylab.plot(range(num_steps), alpha[:, 0], c=cpal[0], label=r"$\hat{\mu}$")
pylab.plot(range(num_steps), mu_ANSWER * np.ones(num_steps), c=cpal[0], linestyle='--', label=r"$\mu$")
pylab.plot(range(num_steps), alpha[:, 1], c=cpal[1], label=r"$\hat{\sigma}$")
pylab.plot(range(num_steps), std_ANSWER * np.ones(num_steps), c=cpal[1], linestyle='--', label=r"$\sigma$")
pylab.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
pylab.xlabel("Iteration")
pylab.ylabel("Hyperparameters")
pylab.show()

