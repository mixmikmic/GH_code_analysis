import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# PSO performance
n_groups = 4
iteration_5 = (0.9389,0.8543,0.8992,0.8367)
iteration_10 = (0.8269,0.8954,0.8956,0.8923)
iteration_40 = (0.8881,0.9069,0.9112,0.9206)
 
# create plot
fig, ax = plt.subplots(figsize=(10, 5))
index = np.arange(0,12,3)
print("index=",index)
bar_width = 0.7
opacity = 0.8
 
rects1 = plt.bar(index, iteration_5, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Maximum Iteration are 5')
 
rects2 = plt.bar(index + bar_width+0.1, iteration_10, bar_width,
                 alpha=opacity,
                 color='y',
                 label='Maximum Iteration are 10')

rects3 = plt.bar(index + bar_width + bar_width+0.2, iteration_40, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Maximum Iteration are 40')

plt.xlabel('Swarm size')
plt.ylabel('Mean Average Error')
plt.title('Particle Swarm Optimization performance')
plt.xticks(index + bar_width + 0.1, ('5', '10', '20', '40'))
plt.legend( bbox_to_anchor=(0.65,1.3))
 
def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.0*height,
                '%.4f' % height,
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
plt.show()

# Linear Regression performance
# PSO performance
n_groups = 4
iteration_5 = (44.76,71.26,77.95,81.78)
 
# create plot
fig, ax = plt.subplots(figsize=(10, 5))
index = np.arange(0,4,1)
print("index=",index)
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.bar(index, iteration_5, bar_width,
                 alpha=opacity,
                 color='g')

plt.xlabel('Minimum number of movies rated by a user')
plt.ylabel('R-squared error')
plt.title('Linear Regression performance')
plt.xticks(index, ('5', '10', '15', '20'))
# plt.legend( bbox_to_anchor=(1.0,1.0))
 
def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.0*height,
                '%.4f' % height,
                ha='center', va='bottom')

autolabel(rects1)
plt.show()

# Comparing performance of all algorithms
n_groups = 4
traditional_UUCF = (3.228,2.9582,2.9282,2.8825,3.3679)
pso = (3.2012,2.9497,2.9442,2.897,2.6359)
linear_regression = (3.3763,3.3587,3.3541,3.4146,3.41399)
 
# create plot
fig, ax = plt.subplots(figsize=(10, 5))
index = np.arange(0,25,5)
print("index=",index)
bar_width = 1.2
opacity = 0.8
 
rects1 = plt.bar(index, traditional_UUCF, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Traditional User-User Collaborative Filtering')
 
rects2 = plt.bar(index + bar_width+0.1, pso, bar_width,
                 alpha=opacity,
                 color='c',
                 label='Particle Swarm Optimization')

rects3 = plt.bar(index + bar_width + bar_width+0.2, linear_regression, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Linear Regression')

plt.xlabel('Neighbourhood size')
plt.ylabel('Mean Average Error')
plt.title('Comparing performance of different prediction models')
plt.xticks(index + bar_width + 0.1, ('5', '10', '15', '20','All'))
plt.legend( bbox_to_anchor=(1.0,1.0))
 
def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.0*height,
                '%.2f' % height,
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
plt.show()

