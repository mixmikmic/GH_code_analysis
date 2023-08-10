import math
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


class Plot:
    def __init__(self, range=(-20,20)):
        self.range = range 
    
    def hist(self, data, color):
        plt.hist(data, 200, range=self.range, normed=True, facecolor=color, alpha=0.2, histtype='stepfilled')
        
    def plot(self, x, y, color, alpha):
        plt.plot(x, y, color=color, alpha=alpha)

    def show(self):
        plt.show() 
        
        
class Gaussian:
    def samples(self, params, samples=2000):
        return [np.random.normal(params[0], params[1]) for x in range(samples)]
            
    def plot_samples(self, plot, params, color='green'):
        plot.hist(self.samples(params), color)
        
    def gaussian(self, x, mean, std):
        return np.exp(-np.power((x - mean) / std, 2) / 2) / (math.sqrt(2 * math.pi) * std)
    
    def make_support(self, plot, scale=100.0):
        return [x / scale for x in range(int(scale * plot.range[0]), int(scale * plot.range[1]))]
        
    def plot_pdf(self, plot, params, color='green', alpha=1):
        support = self.make_support(plot)
        result = [self.gaussian(x, params[0], params[1]) for x in support]
        plot.plot(support, result, color, alpha)
     

plt.figure(figsize=(12,12))

p = Plot()
g = Gaussian()

dist = (0.0, 1.0)

# Distribution histogram in green, with PDF outline
g.plot_samples(p, dist)
g.plot_pdf(p, dist)

p.show() 

plt.figure(figsize=(12,12))

p = Plot()
g1, g2 = Gaussian(), Gaussian()

dist1 = (-5.0, 2.0)
dist2 = (5.0, 0.5)

# First distribution histogram in green, with PDF outline
g1.plot_samples(p, dist1)
g1.plot_pdf(p, dist1)

# Second distribution histogram in red, with PDF outline
g2.plot_samples(p, dist2, 'red')
g2.plot_pdf(p, dist2, 'red')

p.show() 

class Compute:
    def add(self, dist1, dist2):
        mean = dist1[0] + dist2[0]
        std = dist1[1] + dist2[1]
        self.dist = (mean, std)
        
    def multiply(self, dist1, dist2):
        mean = (dist1[0] * dist2[1] + dist1[1] * dist2[0]) / (dist1[1] + dist2[1])
        std = 1 / (1 / dist1[1] + 1 / dist2[1])
        self.dist = (mean, std)
   

plt.figure(figsize=(12,12))

p = Plot()
g = Gaussian()
c = Compute()

dist1 = (-5.0, 2.0)
dist2 = (5.0, 0.5)

# The PDF as a green curve
c.add(dist1, dist2)
g.plot_pdf(p, c.dist)

# Histogram data in green using computed parameters
g.plot_samples(p, c.dist)

p.show() 

plt.figure(figsize=(12,12))

p = Plot()
g, g1, g2 = Gaussian(), Gaussian(), Gaussian()
c = Compute()

dist1 = (-5.0, 2.0)
dist2 = (5.0, 0.5)

# The PDF as a red curve
c.add(dist1, dist2)
g.plot_pdf(p, c.dist, 'red')

# Histogram data in red by actually summing the two priors
s1 = g1.samples(dist1)
s2 = g2.samples(dist2)
g_sum = [s1[i] + s2[i] for i in range(len(s1))]
p.hist(g_sum, 'red')               

p.show() 

plt.figure(figsize=(12,12))

p = Plot()
g = Gaussian()
c = Compute()

dist1 = (-5.0, 1.0)
dist2 = (5.0, 1.0)

# Priors in green
g.plot_pdf(p, dist1, 'green', 0.5)
g.plot_samples(p, dist1, 'green')
g.plot_pdf(p, dist2, 'green', 0.5)
g.plot_samples(p, dist2, 'green')

# Posterior in red
c.multiply(dist1, dist2)
g.plot_pdf(p, c.dist, 'red')
g.plot_samples(p, c.dist, 'red')

p.show() 

plt.figure(figsize=(12,12))

p = Plot()
g = Gaussian()
c = Compute()

dist1 = (-5.0, 1.0)
dist2 = (5.0, 4.0)

# Priors in green
g.plot_pdf(p, dist1, 'green', 0.5)
g.plot_samples(p, dist1, 'green')
g.plot_pdf(p, dist2, 'green', 0.5)
g.plot_samples(p, dist2, 'green')

# Posterior in red
c.multiply(dist1, dist2)
g.plot_pdf(p, c.dist, 'red')
g.plot_samples(p, c.dist, 'red')

p.show() 

plt.figure(figsize=(12,12))

p = Plot()
g = Gaussian()
c = Compute()

dist1 = (0.0, 1.0)
dist2 = (0.0, 8.0)

# Priors in green
g.plot_pdf(p, dist1, 'green', 0.5)
g.plot_samples(p, dist1, 'green')
g.plot_pdf(p, dist2, 'green', 0.5)
g.plot_samples(p, dist2, 'green')

# Posterior in red
c.multiply(dist1, dist2)
g.plot_pdf(p, c.dist, 'red')
g.plot_samples(p, c.dist, 'red')

p.show() 



