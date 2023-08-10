get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt

def approx_pi(n=1000, plot=False):
    points = np.random.rand(n, 2)
    result = np.sqrt(points[:, 0]**2 + points[:, 1]**2) < 1.0
    pi = 4 * (result.sum() /  n)
    if plot:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(points[result, 0], points[result, 1], 'o', color='crimson')
        ax.plot(points[~result, 0], points[~result, 1], 'o', color='steelblue')
        ax.axis('equal')
        ax.axis([0, 1, 0, 1])
    return pi


approx_pi(plot=True)

approx_pi(10**6, plot=True)

for i in range(7):
    print(10**i, approx_pi(10**i))

def approx_times(func, times=10, *args, **kwargs):
    results = []
    for i in range(times):
        results.append(func(*args, **kwargs))
    return np.mean(results)


approx_times(approx_pi, times=3, n=1000)

for i in range(1, 26):
    for j in range(7):
        if i % 5 == 0 and j % 3 == 0:
            print(i, 10**j, approx_times(approx_pi, times=i, n=10**j))
np.pi

