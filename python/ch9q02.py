import numpy as np

a = 1
b = 3
n = 10

a_n = []
b_n = []

for i in range(1000):
    sample = np.random.uniform(low=a, high=b, size=n)
    a_n.append(np.min(sample))
    b_n.append(np.max(sample))
    

a_hat = np.mean(a_n)
a_se = np.std(a_n)
b_hat = np.mean(b_n)
b_se = np.std(b_n)

a_hat, a_hat-1.96*a_se, a_hat+1.96*a_se

b_hat, b_hat-1.96*b_se, b_hat+1.96*b_se



