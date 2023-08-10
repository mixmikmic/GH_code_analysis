import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from mpl_toolkits.mplot3d import Axes3D

house_data = np.genfromtxt('./Datasets/area_br_price.txt', delimiter=',')
X = np.delete(house_data, 2, 1)
X = np.insert(X, 0, 1, axis=1)
Y = np.delete(house_data, [0,1], 1)

def normalizeFeatures (X):
    for i in range(1, np.shape(X)[1]):
        # isolate one of the features and normalize it
        feature = X.T[i]
        feature -= np.average(feature)
        feature /= (np.amax(feature) - np.amin(feature))
        
normalizeFeatures(X)

# Recall that thetaV looked l ike [350000,600000,-50000]
# However we wish ot improve that 
def costFunction (thetaV, X, Y):
    #Yt is the vector with the values that i think are correct
    dataPints = np.shape(Y)[0]
    Yt = np.dot(X, thetaV)
    Yt = Yt.reshape(dataPints, 1) #change shape so it will match Y as a column (T would not work)
    dif = Yt - Y
    sq = np.sqrt(np.square(dif))
    sm = np.sum(sq)
    avg = sm / dataPints
    cost = avg
    return cost # more or less represents how far away we are in avg to the real value

x_min = 570000
x_max = 610000
x_step = 50

y_min = -140000
y_max = -80000
y_step = 50

x = np.arange(x_min, x_max, x_step)
y = np.arange(y_min, y_max, y_step)
X_, Y_ = np.meshgrid(x, y)
zs = np.array([costFunction(np.array([350000, x, y]), X, Y) for x, y in zip(np.ravel(X_), np.ravel(Y_))])
Z = zs.reshape(X_.shape)

plt.figure()
CS = plt.contour(X_, Y_, Z)
plt.clabel(CS, inline=1, fontsize=10)
plt.show()

costFunction(np.array([350000, 590000, -120000]), X, Y)

mini = 300000
maxi = 400000
numSteps = 60 # <<<< Change that parameter
step = maxi/numSteps
iterations = int((maxi - mini)/step)

y = np.empty(iterations, dtype=np.float)
x = np.arange(mini, maxi, step)

for i in range(iterations):
    y[i] = costFunction(np.array([mini + i*step,590000,-120000]), X, Y)

plt.plot(x, y, 'go')
plt.xlabel('thet0')
plt.ylabel('costFunction')
plt.show()

def getPrices (thetaV, X):
    #Yt is the vector with the values that i think are correct
    dataPints = np.shape(X)[0]
    Yt = np.dot(X, thetaV)
    Yt = Yt.reshape(dataPints, 1) #change shape so it will match Y as a column (T would not work)
    return Yt

hypParameters = np.array([340000,590000,-120000])
prices = getPrices (hypParameters, X)

t = PrettyTable(['Expected', 'Real'])

for i in range(np.shape(prices)[0]):
    t.add_row([int(prices[i][0]), int(Y[i][0])])
    
print(t.get_string(start = 0, end = 10))



