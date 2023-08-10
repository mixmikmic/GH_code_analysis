import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

import utils as u
from IPython.core.display import HTML

HTML(u.create_table(pd.read_csv("binary_example.csv")))

def assert_paradox_binary_distribution(p, q):
    """
    Asserts whether a distribution described by p,q demonstrates 
    Simpson's paradox. 
    
    :param p: dictionary, p[x][z] = p(Y=1|X,Z)
    :param q: dictionary, q[x][z] = p(Z|X)
    :return: None
    """
    n_samples = len(p[0])
    for i in range(n_samples):
        assert(p[0][i] < p[1][i])
            
    assert(all(0 < qq < 1 for qq in q[1].values()))
    assert(all(0 < qq < 1 for qq in q[0].values()))
    
    b0 = sum(p[0][i] * q[0][i] for i in range(n_samples))
    b1 = sum(p[1][i] * q[1][i] for i in range(n_samples))
    
    assert(b0 > b1)

from collections import defaultdict

def generate_p_values(n_subgroups):
    """
    Generates a set of conditional probabilities that obey
    
     - p(Y=1|x=1, z) > p(Y=1|x=0, z) for all z
     - p(Y=1|x=1, z=j) > p(Y=1|x=1, k) when j > k
     - p(Y=1|x=1, z=0) < p(Y=1|x=0, z=n) where n = max(z)
     
    :param n_subgroups: int. The number of values $Z$ can take.
    :return: dictionary, p[x][z] = p(Y=1|x,z)
    """
    p = defaultdict(dict)

    boundaries = np.random.uniform(0, 1, size=2 * n_subgroups)
    boundaries = [(n + b) / (2*n_subgroups) for n, b in enumerate(boundaries)]

    for i in range(n_subgroups):
        p[0][i] = boundaries[i*2]
        p[1][i] = boundaries[i*2+1]

    return p

def get_q_weights(ps, target):
    """
    Generates a mixture of the values in ps which is the solution to
    
    \sum_{i} p[i]q[i] = target
    
    :param ps: list of number
    :param target: goal of the sum
    :return: qs: list of weightings of ps
    """
    if len(ps) <= 1:
        raise ValueError("ps cannot be shorter than 2")

    if len(ps) == 2:
        p0, p1 = ps
        q0 = (p1 - target) / (p1 - p0)
        return q0, (1 - q0)

    rest, last = ps[:-1], ps[-1]
    mid_target = np.random.uniform(low=rest[0], high=min(target, rest[-1]))

    q0, q1 = get_q_weights([mid_target, last], target)
    remaining_qs = get_q_weights(rest, mid_target)
    qs = [q0 * q for q in remaining_qs] + [q1]

    return qs

def generate_paradox_binary_distribution(n_subgroups=3):
    """
    Generates a distribution which demonstrates Simpson's paradox 
    
    q[x][z] = p(Z|X)
    p[x][z] = p(Y=1|X,Z)
    
    :param n_subgroups: int
    :return: p, q: dicts
    """

    p = generate_p_values(n_subgroups)
    
    p_low, p_high= p[1][0], p[0][n_subgroups-1]

    b1, b0 = np.random.uniform(size=2)
    b1, b0 = p_low+((1+b1)/5)*(p_high-p_low), p_low+((3+b0)/5)*(p_high-p_low) 
    
    
    q = {}
    q[0] = {k: v for k, v in enumerate(get_q_weights(sorted(p[0].values()), b0))}
    q[1] = {k: v for k, v in enumerate(get_q_weights(sorted(p[1].values()), b1))}

    return p, q

p, q = generate_paradox_binary_distribution(4)

assert_paradox_binary_distribution(p, q)

def realise_binary_paradox(p, q, x, n_approx_samples=100):
    """
    Realises the mean outcome from the provided distribution
    :param p: dict, p[x][z] = p(Y=1|X,Z)
    :param q: dict, q[x][z] = p(Z|X)
    :param x: dict, x[i] = p(x)
    :param n_approx_samples: int
    :return: Dataframe of outcomes
    """
    records = []
    for xx in x.keys():
        for zz in q[xx].keys():
            p_y_eq_0 = x[xx] * q[xx][zz] * (1 - p[xx][zz])
            p_y_eq_1 = x[xx] * q[xx][zz] * p[xx][zz]
            records.append({"x": xx, "z": zz, "y": 1, 
                            "count": round(n_approx_samples * p_y_eq_1)})
            records.append({"x": xx, "z": zz, "y": 0, 
                            "count": round(n_approx_samples * p_y_eq_0)})
    return pd.DataFrame.from_records(records)

p, q = generate_paradox_binary_distribution(n_subgroups=3)
x = {0:0.3, 1:0.7}

df = realise_binary_paradox(p, q, x, n_approx_samples=500)
HTML(u.create_table(df))

def generate_gaussian_simpsons_paradox(n_subgroups=3, n_samples=1000):

    overall_cov = 3*np.array([[1,0.9], [0.9,1]])

    means = np.random.multivariate_normal(mean=[0,0], cov=overall_cov, size=n_subgroups)
    
    weights = np.random.uniform(size=n_subgroups)
    weights /= np.sum(weights)
    covs = [np.random.uniform(0.2,0.8) for _ in range(n_subgroups)]
    covs = [np.array([[1,-c], [-c,1]]) for c in covs]


    samples = []

    for sg, (mean, cov, w) in enumerate(zip(means, covs, weights)):
        n = int(round(n_samples * w))
        sample = np.random.multivariate_normal(mean=mean, cov=cov, size=n)
        sample = pd.DataFrame(sample, columns=["x", "y"])
        sample["z"] = sg
        samples.append(sample)
        
    df = pd.concat(samples)
    
    return df

df = generate_gaussian_simpsons_paradox()

print("Total Covariance: {:.3f}".format(df[["x", "y"]].cov().iloc[0,1]))
for z in df.z.unique():
    print("Subgroup {} covariance: {:.3f}".format(z,df[df.z==z][["x", "y"]].cov().iloc[0,1]))

sns.regplot(data=df, x="x", y="y");

fig, ax = plt.subplots()

plt.xlim(-6,6);
plt.ylim(-6,6);

for z in df.z.unique():
    sns.regplot(data=df[df.z==z], x="x", y="y", ax=ax)

