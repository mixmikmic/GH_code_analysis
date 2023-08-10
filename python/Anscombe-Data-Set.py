import matplotlib.pyplot as plt
import numpy as np
import pandas
import statistics
from statistics import variance
from pylab import *
from collections import OrderedDict

x1 = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],dtype= float)
y1 = np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])
y2 = np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74])
y3 = np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73])
x4 = np.array([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],dtype= float)
y4 = np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89])

data_set = [('Set I_x',x1),
         ('Set I_y',y1),
         ('Set II_x',x1),
         ("Set II_y",y2),
         ("Set III_x",x1),
         ('Set III_y', y3),
         ("Set IV_x",x4),
         ('Set IV_y',y4)]
data_set = OrderedDict(data_set)
pandas.DataFrame(data_set)

# Calculate statistics for each set
metric_list = ["Mean of x", "Mean of y", "Variance of x","Variance of y","Pearson product-moment correlation(x,y) "]
set_list = ["Set I", "Set II", "Set III","Set IV"]
data = np.array([[mean(x1),mean(y1) , variance(x1),variance(y1), np.corrcoef(x1,y1)[1,0]  ],
                 [mean(x1),mean(y2) , variance(x1), variance(y2), np.corrcoef(x1,y2)[1,0] ],
                 [mean(x1), mean(y3), variance(x1),variance(y3), np.corrcoef(x1,y3)[1,0]  ],
                 [mean(x4), mean(y4), variance(x4),variance(y4), np.corrcoef(x4,y4)[1,0]  ] ])


pandas.DataFrame(data, set_list, metric_list)

fig = plt.figure(figsize=(15,10))

# Set I
ax1 = fig.add_subplot(221)
ax1.scatter(x1, y1, c='orangered',edgecolors= 'orangered')
m,b = np.polyfit(x1, y1, 1)
X = np.linspace(ax1.get_xlim()[0], ax1.get_xlim()[1], 100)
ax1.set_title("Set I")
ax1.plot(X, m*X+b, '-')

# Set II
ax2 = fig.add_subplot(222)
ax2.scatter(x1, y2, c='orangered',edgecolors= 'orangered')
m,b = np.polyfit(x1, y1, 1)
X = np.linspace(ax2.get_xlim()[0], ax2.get_xlim()[1], 100)
ax2.set_title("Set II")
ax2.plot(X, m*X+b, '-')

# Set III
ax3 = fig.add_subplot(223)
ax3.scatter(x1, y3, c='orangered',edgecolors= 'orangered')
m,b = np.polyfit(x1, y3, 1)
X = np.linspace(ax3.get_xlim()[0], ax3.get_xlim()[1], 100)
ax3.set_title("Set III")
ax3.plot(X, m*X+b, '-')
        
# Set IV
ax4 = fig.add_subplot(224)
ax4.scatter(x4, y4, c='orangered',edgecolors= 'orangered')
m,b = np.polyfit(x4, y4, 1)
X = np.linspace(ax4.get_xlim()[0], ax4.get_xlim()[1], 100)
ax4.set_title("Set IV")
ax4.plot(X, m*X+b, '-')

plt.show()



