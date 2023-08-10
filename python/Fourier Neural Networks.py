import numpy as np
import matplotlib.pyplot as plt

ys = []
xs = []
x2 = []
for n in range(1,250):
    s = len(set([(i-1)*(j-1) for i in range(n) for j in range(n)]))
    ys.append(s)
    xs.append(n)
    x2.append(n**2/250)
    
plt.plot(xs, ys)
plt.plot(x2, ys)
plt.show()





