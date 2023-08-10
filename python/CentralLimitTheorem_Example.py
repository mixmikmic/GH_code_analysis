import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

MeanExpected = 3.5
DevExpected = np.sqrt( 1/6 * sum( [(i-3.5)**2 for i in np.arange(1,7)] ) )
print('We expect mean {} with deviation {}.'.format(MeanExpected, round(DevExpected, 2)))

# Roll several dice and return sum of result
def roll(nDice):
    dice = np.random.choice(np.arange(1,7),nDice)
    return sum(dice)

# For example - rolling 2 dice 10 times:
[roll(2) for i in np.arange(10)]

# Creating a large data set - rolling 6 dice 10,000 times !
nDice = 6
data = np.array([roll(nDice) for i in range(10000)])

# Plot a histogram
plt.figure(num=None, figsize=(6, 4), dpi=100)
n, bins, patches = plt.hist( data, 
                             bins=np.arange(nDice-0.5,nDice*6+1.5,1), 
                             normed=1, 
                             facecolor='blue', 
                             alpha=0.75 )

# Gaussian fit - using theoretical Mean and Dev
from scipy.stats import norm
import matplotlib.mlab as mlab

y = mlab.normpdf(bins, nDice*MeanExpected, np.sqrt(nDice)*DevExpected)
plt.plot(bins, y, 'r--', linewidth=2)

plt.xlabel('Distribution of sums')
plt.ylabel('Frequency (probability to observe)')
plt.title('Histogram of '+str(nDice)+' dice rolls')
plt.show()



