#Import Statements
import numpy as np
import matplotlib.pyplot as plt

#Collect Data
my_data = np.genfromtxt('../ML/mnist_test.csv', delimiter=',')
train = my_data[0:1999]
test = my_data[2000:2100]

#Functions
def getCharacter(char, side=28):
    size = side * side
    nm = char[0]
    pxs = np.zeros(size)
    for i in range(1, size + 1):
        pxs[i-1] = char[i]
    pxs = np.reshape(pxs, (side, side))
    pxs = np.fliplr([pxs])[0]
    
    return nm, pxs

def plotCharacter(pxs):
    side = pxs.shape[0]
    y = x = range(1, side + 1)
    x, y = np.meshgrid(x, y)
    plt.pcolormesh(x, y, pxs, cmap=plt.cm.get_cmap('Greys'))
    plt.colorbar()
    plt.show()
    
def findDistance(pxs1, pxs2, square=False):
    if square:
        return np.sum(np.square(pxs1 - pxs2))
    else:
        return np.sum(np.absolute(pxs1 - pxs2))

def nearestNeighbour( train, pxs ):
    #Give a default value to minDif to then compare and minimize
    _, iniChar = getCharacter(train[0])
    nearest = train[0]
    minDif = findDistance(iniChar, pxs)
    #Iterate over the other chars to check which is the single best
    for char in train:
        _, pxs2 = getCharacter(char)
        dif = findDistance(pxs2, pxs)        
        if dif < minDif:
            minDif = dif
            nearest = char
            
    return nearest

def calculateAccuracy(train, test):
    hits = 0
    
    for char in test:
        nm, pxs = getCharacter(char)
        nearest = nearestNeighbour(train, pxs)
        value, _ = getCharacter(nearest)
        
        if value == nm:
            hits += 1
    
    return 100*(hits/test.shape[0])

print("The accuracy of your algorithm is,",calculateAccuracy(train, test), "%")



