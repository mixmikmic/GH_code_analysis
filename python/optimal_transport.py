import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')

def _normalise(source, target):
    return np.divide(source, np.sum(source))

def _optimaltransport(source, target):

    # normalise densities to have equal sum. Integers for ease.
    
    if len(source) == 0:
        mapping, cost = np.zeros((1,1)), 0.000001
        return mapping, cost
    
    source, target = np.array(source), np.array(target)
    f_x, g_y = _normalise(source, target), _normalise(target, source)
    
    if len(f_x) == 1:
        m, n = 100000000, len(g_y)
    else:        
        m, n = len(f_x), len(g_y)
       
    c, i, j = 0, 0, 0
    
    mapping = np.zeros((m, n)) # Can create heatmap to visualise mapping. Only for small m, n! Or use sparse matrix

    while i < m and j < n:
        if g_y[j] == 0: 
            j += 1
        elif f_x[i] == 0: # if supply/demand if empty, skip. 
            i += 1
        else:
            if f_x[i] - g_y[j] > 0:
                f_x[i] -= g_y[j]
                c += (i/(m-1) - j/(n-1)) ** 2 * g_y[j] # density * cost to transport
                mapping[i,j] = g_y[j]
                j += 1
            elif f_x[i] - g_y[j] < 0:
                g_y[j] -= f_x[i]
                c += (i/(m-1) - j/(n-1)) ** 2 * f_x[i] # density * cost to transport
                mapping[i,j] = f_x[i]
                i += 1
            else: 
                c += (i/(m-1) - j/(n-1)) ** 2 * f_x[i] # density * cost to transport
                mapping[i,j] = f_x[i]
                i += 1                
                j += 1
    
    return mapping, c

# set up densities

k = 4

m, n = 2**k, 2**k

source = 2*np.array(list(range(m))) / m 
target = np.ones(n)

plt.plot(source, color = 'r')
plt.plot(target, color = 'b')

mapping, cost = _optimaltransport(source, target)

plt.gcf().set_size_inches(18,12)
plt.gca().set_aspect('equal')
print('\n' + 'Transport cost: ' + str(cost) + '\n')
sns.heatmap(mapping)

# set up densities

for i in range(1,10):

    m, n = 2**i, 2**i

    source = 2*np.array(list(range(m))) / m 
    target = np.ones(n)
    
    mapping, cost = _optimaltransport(source, target)
    
    print('Cost: ' + str(cost))
    print('Error: ' + str(1/30 - cost) + '\n')

plt.gcf().set_size_inches(18,12)
plt.gca().set_aspect('equal')
print('\n' + 'Transport cost: ' + str(cost) + '\n')
sns.heatmap(mapping)

# set up densities

k = 4

m, n = 2**k, 2**k

source = 2*np.array(list(range(m))) / m 
target = 2*np.array(list(reversed(range(n)))) / n 

plt.plot(source, color='r')
plt.plot(target, color='b')

mapping, cost = _optimaltransport(source, target)

plt.gcf().set_size_inches(18,12)
plt.gca().set_aspect('equal')
print('\n' + 'Transport cost: ' + str(cost) + '\n')
sns.heatmap(mapping)

# set up densities

for i in range(1,10):

    m, n = 2**i, 2**i

    source = 2*np.array(list(range(m))) / m 
    target = 2*np.array(list(reversed(range(n)))) / n 
    
    mapping, cost = _optimaltransport(source, target)
    
    print('Cost: ' + str(cost))
    print('Error: ' + str(1/30 - cost) + '\n')

plt.gcf().set_size_inches(18,12)
plt.gca().set_aspect('equal')
print('\n' + 'Transport cost: ' + str(cost) + '\n')
sns.heatmap(mapping)



