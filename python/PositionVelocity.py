import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


class Plot:
    def __init__(self, limits=((-10,10),(-10,10))):
        x, y = limits
        x_min, x_max = x
        y_min, y_max = y
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
    def scatter(self, data, color='g', alpha=0.5, area=20):
        x = [d[0] for d in data]
        y = [d[1] for d in data]
        areas = [area for d in data]
        colors = [color for d in data]
        plt.scatter(x, y, s=areas, c=colors, alpha=alpha, edgecolor=None)
        
    def show(self):
        plt.show() 
        

class Gaussian:
    def generate(self, params, samples=300):
        self.data = []
        for i in range(samples):
            self.data.append(np.random.multivariate_normal(params[0], params[1]))
    
    def points(self):
        return self.data

plt.figure(figsize=(12,12))

p = Plot(((-5,5),(-5,5)))            
            
# Position estimate 
g = Gaussian()
mean = [-2, 0]
cov = [[0.1**2, 0], [0, 3**2]]
g.generate((mean, cov))
p.scatter(g.points(), 'y')      

p.show()    
    

plt.figure(figsize=(12,12))

p = Plot(((-5,5),(-5,5)))            
            
# Position estimate 
g = Gaussian()
mean = [-2, 0]
cov = [[0.1**2, 0], [0.0, 3**2]]
g.generate((mean, cov))
p.scatter(g.points(), 'y', 0.25)     

# Projection of all possible positions and velicities given assumptions
g = Gaussian()
mean = [-2, 0]
cov = [[4.02, 4], [4, 4.04]]
g.generate((mean, cov))
p.scatter(g.points(), 'g')    

p.show()    
    

plt.figure(figsize=(12,12))

p = Plot(((-5,5),(-5,5)))            
            
# Position estimate 
g = Gaussian()
mean = [-2, 0]
cov = [[0.1**2, 0], [0.0, 3**2]]
g.generate((mean, cov))
p.scatter(g.points(), 'y', 0.1)     

# Projection of all possible positions and velicities given assumptions
g = Gaussian()
mean = [-2, 0]
cov = [[4.02, 4], [4, 4.04]]
g.generate((mean, cov))
p.scatter(g.points(), 'g', 0.3)   

# New position estimate
g = Gaussian()
mean = [0, 0]
cov = [[0.1**2, 0], [0.0, 3**2]]
g.generate((mean, cov))
p.scatter(g.points(), 'b')    

p.show()    
    

plt.figure(figsize=(12,12))

p = Plot(((-5,5),(-5,5)))            

# Position estimate 
g = Gaussian()
mean = [-2, 0]
cov = [[0.1**2, 0], [0.0, 3**2]]
g.generate((mean, cov))
p.scatter(g.points(), 'y', 0.1)     

# Projection of all possible positions and velicities given assumptions
g = Gaussian()
mean = [-2, 0]
cov = [[4.02, 4], [4, 4.04]]
g.generate((mean, cov))
p.scatter(g.points(), 'g', 0.1)   

# New position estimate
g = Gaussian()
mean = [0, 0]
cov = [[0.1**2, 0], [0.0, 3**2]]
g.generate((mean, cov))
p.scatter(g.points(), 'b', 0.1)  

# Composite estimate
g = Gaussian()
mean = [0, 2]
cov = [[0.2**2, 0.02], [0.02, 0.2**2]]
g.generate((mean, cov))
p.scatter(g.points(), 'r') 

p.show()    



