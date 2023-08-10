import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

n = 100

sample = []
cl = []

p = 0.3
mu0 = 0
mu1 = 10

# There is probably a better way to do this...
for i in range(n):
    u = np.random.rand();
    if (u <= p):
        sample.append(np.random.normal(mu0,1));
        cl.append(0)
    else:
        sample.append(np.random.normal(mu1,1));
        cl.append(1)
        
data = pd.DataFrame(list(zip(sample,cl)), columns = ['Sample', 'Type'])

data.head()

plt.figure(figsize=(10,8))
type1_data = data[data.Type == 1]
type0_data = data[data.Type == 0]
plt.scatter(type1_data.Sample, np.zeros(len(type1_data)), s = 200, c = 'r');
plt.scatter(type0_data.Sample, np.ones(len(type0_data)), s = 200, c = 'b');

plt.show()

from scipy import stats

def tau(p, mu0, mu1, data):
    # Computes the tau-factors
    a = stats.norm.pdf(data, loc=mu0, scale=1);
    b = stats.norm.pdf(data, loc=mu1, scale=1);
    
    return p * a / (p * a + (1-p) * b)
    
def update(p_old, mu0_old, mu1_old, y_data):
    # Update rule for the iteration
    tau_factors = tau(p_old, mu0_old, mu1_old, y_data);
    p_new = tau_factors.mean()
    mu0_new = np.dot(y_data, 1-tau_factors) / (1-tau_factors).sum()
    mu1_new = np.dot(y_data, tau_factors) / tau_factors.sum()
    return [p_new, mu0_new, mu1_new]

initial_theta = [0.5, 2, 8]

num_of_iter = 20

theta = initial_theta
print(theta)

for i in range(num_of_iter):
    theta = update(theta[0], theta[1], theta[2], data.Sample)
    print(theta)

n = 1000

sample = []
cl = []

p = 0.3
mu0 = 0
mu1 = 2

# There is probably a better way to do this...
for i in range(n):
    u = np.random.rand();
    if (u <= p):
        sample.append(np.random.normal(mu0,1));
        cl.append(0)
    else:
        sample.append(np.random.normal(mu1,1));
        cl.append(1)
        
data = pd.DataFrame(list(zip(sample,cl)), columns = ['Sample', 'Type'])

plt.figure(figsize=(10,8))
type1_data = data[data.Type == 1]
type0_data = data[data.Type == 0]
plt.scatter(type1_data.Sample, np.zeros(len(type1_data)), s = 200, c = 'r');
plt.scatter(type0_data.Sample, np.ones(len(type0_data)), s = 200, c = 'b');

plt.show()

initial_theta = [0.3, 0, 2]

num_of_iter = 10

theta = initial_theta
print(theta)

for i in range(num_of_iter):
    theta = update(theta[0], theta[1], theta[2], data.Sample)
    print(theta)

