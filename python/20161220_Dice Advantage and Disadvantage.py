import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
#Remember that Python is zero-indexed, and the range function will return 
#up to one value less than the second parameter
roll1poss = list(range(1,21))
roll2poss = list(range(1,21))
#This next line might look scary, but we are finding the maximum of (each 
#first roll out of all possible first rolls) and (each second roll out of 
#all possible second rolls), using a listcomp
maxRoll = [max(roll1,roll2) for roll1 in roll1poss for roll2 in roll2poss]
#Now, we want to reshape the array so that it is 20x20 instead of 1x400
maxRollReshape = np.reshape(maxRoll,[20,20])
#Finally, let's plot the results. I wrote it as a function so it can be
#re-used for Case 2, as well
def plotHeatmap(rollResults, titleLab):
    f, ax = plt.subplots(figsize=(8, 6))
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    sns.heatmap(rollResults, cmap=cmap, vmax=20, square=True, 
                linewidths=.5, cbar_kws={"shrink": .5}, ax=ax,
                xticklabels=roll1poss,yticklabels=roll2poss);
    ax.set_xlabel('First Roll')
    ax.set_ylabel('Second Roll')
    ax.set_title(titleLab)
plotHeatmap(maxRollReshape,'Case 1: Maximum of Two 20-Sided Dice');

#Using the same reasoning as in the previous code
minRoll = [min(roll1,roll2) for roll1 in roll1poss for roll2 in roll2poss]
minRollReshape = np.reshape(minRoll,[20,20])
plotHeatmap(minRollReshape,'Case 2: Minimum of Two 20-Sided Dice');

#Sum all the possible results, then divide by the number of elements
maxExpect = sum(maxRoll)/len(maxRoll)
minExpect = sum(minRoll)/len(minRoll)
normExpect = sum(roll1poss)/len(roll1poss)
stdExpect = np.std(roll1poss)
print("The expected value of Case 1 (Advantage) is:",maxExpect)
print("The expected value of Case 2 (Disadvantage) is:",minExpect)
print("The expected value of a normal 20-sided dice is:",normExpect)

#Set up a side-by-side plot with two axes
f,((ax1,ax2)) = plt.subplots(1,2,figsize=(8,6)) 
#Define this as a function so I don't have to repeat myself for both plots
def plotHistogram(curAx, listVal, titleLab, scaling):
    curAx.axis([1,20,0,scaling])
    values = np.array(listVal)
    sns.distplot(values,ax=curAx, kde=False, bins=20) 
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textStr = '$\mu=%.3f$\n$\sigma=%.3f$' % (values.mean(), values.std())
    curAx.text(0.05, 0.95, textStr, transform=curAx.transAxes, fontsize=14,
               verticalalignment='top', bbox=props)
    curAx.set_title('Histogram of {}'.format(titleLab))
    curAx.set_ylabel('# Occurences')
    curAx.set_xlabel('{} Value'.format(titleLab))    
plotHistogram(ax1, maxRoll, 'Case 1 (Maximum)',50)
plotHistogram(ax2, minRoll, 'Case 2 (Minimum)',50)

from random import randint
#The numHist variable specifies how many times we will be rolling 
#the dice. More is better, but slower.
numHist = 1000000
#Case 1- Maximum of 2 dice rolls
resultListMax = []
for ii in range(numHist):
    roll1 = randint(1,20)
    roll2 = randint(1,20)
    resultListMax.append(max(roll1,roll2))
#Case 2- Minimium of 2 dice rolls
resultListMin = []
for ii in range(numHist):
    roll1 = randint(1,20)
    roll2 = randint(1,20)
    resultListMin.append(min(roll1,roll2))
#Visualize the results
f,((ax1,ax2)) = plt.subplots(1,2,figsize=(8,6)) 
plotHistogram(ax1, resultListMax, 'Case 1 (Maximum)',150000)
plotHistogram(ax2, resultListMin, 'Case 2 (Minimum)',150000)

