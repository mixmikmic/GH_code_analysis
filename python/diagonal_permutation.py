get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import csv, random
import numpy as np
from scipy import spatial

pcafields = ['PC' + str(x) for x in range(1,15)]
# Here we just create a list of strings that will
# correspond to field names in the data provided
# by Mauch et al.

pca = list()
with open('nb/quarterlypca.csv', encoding = 'utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        newpcarow = []
        for field in pcafields:
            newpcarow.append(float(row[field]))
        pca.append(newpcarow)

pca = np.array(pca)
print(pca.shape)

def distance_matrix(pca):
    observations, dimensions = pca.shape
    distmat = np.zeros((observations, observations))
    for i in range(observations):
        for j in range(observations):
            dist = spatial.distance.cosine(pca[i], pca[j])
            distmat[i, j] = dist
    return distmat

d = distance_matrix(pca)    

plt.rcParams["figure.figsize"] = [9.0, 6.0]
plt.matshow(d, origin = 'lower', cmap = plt.cm.YlOrRd, extent = [1960, 2010, 1960, 2010])
plt.show()

def make_foote(quart):
    tophalf = [-1] * quart + [1] * quart
    bottomhalf = [1] * quart + [-1] * quart
    foote = list()
    for i in range(quart):
        foote.append(tophalf)
    for i in range(quart):
        foote.append(bottomhalf)
    foote = np.array(foote)
    return foote

foote5 = make_foote(20)
# This gives us a Foote matrix with a five-year half-width.
# 5 becomes 20 because the underlying dataset has four
# "quarters" of data in each year.

def foote_novelty(distmat, foote):
    axis1, axis2 = distmat.shape
    assert axis1 == axis2
    distsize = axis1
    
    axis1, axis2 = foote.shape
    assert axis1 == axis2
    halfwidth = axis1 / 2
    
    novelties = []
    
    for i in range(distsize):
        
        start = int(i - halfwidth)
        end = int(i + halfwidth)
        
        if start < 0 or end > (distsize - 1):
            novelties.append(0)
        else:
            novelties.append(np.sum(foote * distmat[start: end, start: end]))
    
    return novelties

def getyears():
    years = []
    for i in range(200):
        years.append(1960 + i*0.25)
    return years

years = getyears()

novelties = foote_novelty(d, foote5)
plt.plot(years, novelties)
plt.show()
print("Max novelty for a five-year half-width: " + str(np.max(novelties)))

randomized = np.array(pca)
np.random.shuffle(randomized)
randdist = distance_matrix(randomized)
plt.matshow(randdist, origin = 'lower', cmap = plt.cm.YlOrRd, extent = [1960, 2010, 1960, 2010])
plt.show()

def diagonal_permute(d):
    newmat = np.zeros((200, 200))
    
    # We create one randomly-permuted list of integers called "translate"
    # that is going to be used for the whole matrix.
    
    translate = [i for i in range(200)]
    random.shuffle(translate)
    
    # Because distances matrices are symmetrical, we're going to be doing
    # two diagonals at once each time. We only need one set of values
    # (because symmetrical) but we need two sets of indices in the original
    # matrix so we know where to put the values back when we're done permuting
    # them.
    
    for i in range(0, 200):
        indices1 = []
        indices2 = []
        values = []
        for x in range(200):
            y1 = x + i
            y2 = x - i
            if y1 >= 0 and y1 < 200:
                values.append(d[x, y1])
                indices1.append((x, y1))
            if y2 >= 0 and y2 < 200:
                indices2.append((x, y2))
        
        # Okay, for each diagonal, we permute the values.
        # We'll store the permuted values in newvalues.
        # We also check to see how many values we have,
        # so we can randomly select values if needed.
        
        newvalues = []
        lenvals = len(values)
        vallist = [i for i in range(lenvals)]
        
        for indexes, value in zip(indices1, values):
            x, y = indexes
            
            xposition = translate[x]
            yposition = translate[y]
            
            # We're going to key the randomization to the x, y
            # values for each point, insofar as that's possible.
            # Doing this will ensure that specific horizontal and
            # vertical lines preserve the dependence relations in
            # the original matrix.
            
            # But the way we're doing this is to use the permuted
            # x (or y) values to select an index in our list of
            # values in the present diagonal, and that's only possible
            # if the list is long enough to permit it. So we check:
            
            if xposition < 0 and yposition < 0:
                position = random.choice(vallist)
            elif xposition >= lenvals and yposition >= lenvals:
                position = random.choice(vallist)
            elif xposition < 0:
                position = yposition
            elif yposition < 0:
                position = xposition
            elif xposition >= lenvals:
                position = yposition
            elif yposition >= lenvals:
                position = xposition
            else:
                position = random.choice([xposition, yposition])
                # If either x or y could be used as an index, we
                # select randomly.
            
            # Whatever index was chosen, we use it to select a value
            # from our diagonal. 
            
            newvalues.append(values[position])
            
        values = newvalues
        
        # Now we lay down (both versions of) the diagonal in the
        # new matrix.
        
        for idxtuple1, idxtuple2, value in zip(indices1, indices2, values):
            x, y = idxtuple1
            newmat[x, y] = value
            x, y = idxtuple2
            newmat[x, y] = value
    
    return newmat

newmat = diagonal_permute(d)
plt.matshow(newmat, origin = 'lower', cmap = plt.cm.YlOrRd, extent = [1960, 2010, 1960, 2010])
plt.show()

novelties = foote_novelty(newmat, foote5)
years = getyears()
plt.plot(years, novelties)
plt.show()
print("Max novelty for five-year half-width:" + str(np.max(novelties)))

def zeroless(sequence):
    newseq = []
    for element in sequence:
        if element > 0.01:
            newseq.append(element)
    return newseq
print("Min novelty for five-year half-width:" + str(np.min(zeroless(novelties))))

def permute_test(distmatrix, yrwidth):
    footematrix = make_foote(4 * yrwidth)
    actual_novelties = foote_novelty(distmatrix, footematrix)
    
    permuted_peaks = []
    permuted_troughs = []
    for i in range(100):
        randdist = diagonal_permute(distmatrix)
        nov = foote_novelty(randdist, footematrix)
        nov = zeroless(nov)
        permuted_peaks.append(np.max(nov))
        permuted_troughs.append(np.min(nov))
    permuted_peaks.sort(reverse = True)
    permuted_troughs.sort(reverse = True)
    threshold05 = permuted_peaks[4]
    threshold01 = permuted_peaks[0]
    threshold95 = permuted_troughs[94]
    threshold99 = permuted_troughs[99]
    print(threshold01)
    print(threshold99)
    
    significance = np.ones(len(actual_novelties))
    for idx, novelty in enumerate(actual_novelties):

        if novelty > threshold05 or novelty < threshold95:
            significance[idx] = 0.049
        if novelty > threshold01 or novelty < threshold99:
            significance[idx] = 0.009
    
    return actual_novelties, significance, threshold01, threshold05, threshold95, threshold99

def colored_segments(novelties, significance):
    x = []
    y = []
    t = []
    idx = 0
    for nov, sig in zip(novelties, significance):
        if nov > 1:
            x.append(idx/4 + 1960)
            y.append(nov)
            t.append(sig)
        idx += 1
        
    x = np.array(x)
    y = np.array(y)
    t = np.array(t)
    
    points = np.array([x,y]).transpose().reshape(-1,1,2)
    segs = np.concatenate([points[:-1],points[1:]],axis=1)
    lc = LineCollection(segs, cmap=plt.get_cmap('jet'))
    lc.set_array(t)
    
    return lc, x, y

novelties, significance, threshold01, threshold05, threshold95, threshold99 = permute_test(d, 5)
years = []
for i in range(200):
    years.append(1960 + i*0.25)
plt.plot(years, novelties)
startpoint = years[0]
endpoint = years[199]
plt.hlines(threshold05, startpoint, endpoint, 'r', linewidth = 3)
plt.hlines(threshold95, startpoint, endpoint, 'r', linewidth = 3)
plt.show()

lc, x, y = colored_segments(novelties, significance)

plt.gca().add_collection(lc) # add the collection to the plot
plt.xlim(1960, 2010) # line collections don't auto-scale the plot
plt.ylim(y.min(), y.max())
plt.show()
    

def zeroless_seq(thefilter, filtereda, filteredb):
    thefilter = np.array(thefilter)
    filtereda = np.array(filtereda)
    filteredb = np.array(filteredb)
    filtereda = filtereda[thefilter > 0]
    filteredb = filteredb[thefilter > 0]
    thefilter = thefilter[thefilter > 0]
    return thefilter, filtereda, filteredb

plt.clf()
plt.axis([1960, 2010, 45, 325])
novelties, significance, threshold01, threshold05, threshold95, threshold99 = permute_test(d, 5)
novelties, years, significance = zeroless_seq(novelties, getyears(), significance)
yplot = novelties[significance < 0.05]
xplot = years[significance < 0.05]
plt.scatter(xplot, yplot, c = 'red')
plt.plot(years, novelties)
years = getyears()
startpoint = years[0]
endpoint = years[199]
plt.hlines(threshold05, startpoint, endpoint, 'r', linewidth = 3)
plt.hlines(threshold95, startpoint, endpoint, 'r', linewidth = 3)
plt.show()

def pacechange(startdate, enddate, pca):
    years = getyears()
    startidx = years.index(startdate)
    endidx = years.index(enddate)
    midpoint = int((startidx + endidx)/2)
    firsthalf = np.zeros(14)
    for i in range(startidx,midpoint):
        firsthalf = firsthalf + pca[i]
    secondhalf = np.zeros(14)
    for i in range(midpoint, endidx):
        secondhalf = secondhalf + pca[i]
    return spatial.distance.cosine(firsthalf, secondhalf)

print(pacechange(1990, 1994, pca))
print(pacechange(2001, 2005, pca))

thesum = 0
theobservations = 0
for i in range(1960, 2006):
    theobservations += 1
    thesum += pacechange(i, i+4, pca)
print(thesum / theobservations)

plt.axis([1960, 2010, 0, y.max() + 10])
    
def add_scatter(d, width):
    novelties, significance, threshold01, threshold05, threshold95, threshold99 = permute_test(d, width)
    novelties, years, significance = zeroless_seq(novelties, getyears(), significance)
    yplot = novelties[significance < 0.05]
    xplot = years[significance < 0.05]
    plt.scatter(xplot, yplot, c = 'red')
    plt.plot(years, novelties)

add_scatter(d, 3)
add_scatter(d, 4)
add_scatter(d, 5)
plt.ylabel('Foote novelty')
plt.show()





