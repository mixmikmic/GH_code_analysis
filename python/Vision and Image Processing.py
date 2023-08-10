get_ipython().magic('matplotlib inline')

import numpy as np
import scipy.ndimage as sci
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})

color = sci.imread('data/img/Obama2k11SOTU.jpg')
print(np.shape(color))

plt.imshow(color)

nx = np.shape(color)[0]
ny = np.shape(color)[1]
gray = np.empty(shape=(nx,ny))
print(np.shape(gray))

def coloravg(pixel):
    #return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]
    return np.mean([pixel[i] for i in range(3)])

for x in range(nx):
    for y in range(ny):
        gray[x][y] = coloravg(color[x][y])

cmap_gray = plt.cm.Greys_r
plt.imshow(gray, cmap=cmap_gray)

plt.plot(gray[int(425/2),:])
plt.show()

def gaussian(x, mu, sig):
    return (1/(np.sqrt(2*np.pi*sig*sig))) * np.exp(-np.power(x - mu, 2.) / (2 * np.pi * np.power(sig, 2.)))

for sigma in np.logspace(-1,1,3):
    print("-"*100)
    print("Sigma = {0}".format(sigma))
    
    fig = plt.figure(figsize=(14,5))
    ax1, ax2 = [fig.add_subplot(121+i) for i in range(2)]
    
    result = sci.filters.gaussian_filter(gray,sigma)
    ax1.imshow(result, cmap=cmap_gray)

    xx = np.linspace(-5,5,120)
    ax2.plot(xx, gaussian(xx,0,sigma))    

    plt.show()

for sigma in np.logspace(-1,1,3):
    print("-"*100)
    print("Sigma = {0}".format(sigma))
    
    fig = plt.figure(figsize=(14,5))
    ax1, ax2, ax3 = [fig.add_subplot(131+i) for i in range(3)]
    
    filtered = sci.filters.gaussian_filter(gray,sigma)
    ax1.imshow(filtered, cmap=cmap_gray)
    
    laplace = sci.filters.laplace(filtered)
    ax2.imshow(laplace, cmap=cmap_gray)
    ax3.imshow(np.abs(laplace), cmap=cmap_gray)
    
    plt.show()



