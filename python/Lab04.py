# shows the bins we want to by adding: bins = 6,range = (40,100) to hist()

import numpy as np
import matplotlib.pyplot as plt

x = [88, 48, 60, 51, 57, 85, 69, 75, 97, 72, 71, 79, 65, 63, 73]


#plt.hist returns a triple:
#1) the first (n_d below): the count in each bin
#2) the second (bins_d below): the bins interval
#3) the third(dummy): a python structure that we don't need for now


plt.figure(figsize = (10,10)) # adjust figure size

plt.subplot(211)
n_d, bins_d, dummy = plt.hist(x,bins = 6, range = (40,100),
                              edgecolor='black', linewidth=1.2)

plt.subplot(212)
n_d, bins_d, dummy = plt.hist(x,bins = 6*2, range = (40,100),
                              edgecolor='black', linewidth=1.2)
#
#plt.plot(bins_d,np.zeros(len(bins_d)),'y*')

plt.show()

print(n_d,)
print(bins_d)

# shows the bins we want to by adding: bins = 6,range = (40,100) to hist()

import numpy as np
import matplotlib.pyplot as plt

x = [88, 48, 60, 51, 57, 85, 69, 75, 97, 72, 71, 79, 65, 63, 73]


#plt.hist returns a triple:
#1) the first (n_d below): the count in each bin
#2) the second (bins_d below): the bins interval
#3) the third(dummy): a python structure that we don't need for now


plt.figure(figsize = (8,5)) # adjust figure size

# inside plt.hist, use normed = 1 to get a density histogram
n_d, bins_d, dummy = plt.hist(x,bins = 6, range = (40,100),normed = 1,
                              edgecolor='black', linewidth=1.2)

#
#plt.plot(bins_d,np.zeros(len(bins_d)),'y*')

plt.show()

print(n_d,)
print(bins_d)

import matplotlib.pyplot as plt
import numpy as np

# [0,1] 2 points, including 0,1
pts1 = np.linspace(0,1,2)

# [0,1] 3 points, including 0,1
pts2 = np.linspace(0,1,3)

print('np.linspace(0,1,2)',pts1)
print('np.linspace(0,1,4)',pts2)


# 1-d array with 100 elements from 0 to 4pi, with step 4pi/10
# linspace specifies: 100 points evenly spaced  [0, 4*pi] inclusive
# note the distance between adjacent points is 4*np.pi/99

xpts = np.linspace(0,4*np.pi,100)


# shape '*', '.', '-'
plt.figure(figsize=(10,5))
plt.plot(xpts,'.',color = 'red')

#plt.xlabel("ith value")
#plt.ylabel('linspace')
#plt.title('linspace')

#plt.style.use('ggplot')
plt.show()


xpts = np.linspace(0,4*np.pi,100)

# y is sin of x
y = np.sin(xpts)

# plot the y = sin(x) function where x is between 0 and 4pi
# alpha is between 0 and 1, this is a value changes transparency
# '.' is for shape, you can also have '.-' just try different shapes
plt.plot(xpts,y,'.',label = u'y=sin(x)')#,color = 'blue',alpha = 0.6)

# add a horizontal line x=0
plt.axhline(0, c='black')

# add a vertical line x=2pi
plt.axvline(2*np.pi, c='red',label = u'x = 2pi')

plt.xlabel("x")
plt.ylabel("sin(x)")

plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# define exponential distribution
pdf_exp = lambda x, lam: lam*np.exp(-lam*x)*(x>0)

# the above is the same as the following def function
def pdf_exp2(x,lam):
    return lam*np.exp(-lam*x)*(x>0)


# note that both functions work for array x
# but do not work for list x
# will not compile if you assign x = [1,2,3]
x = np.array([1,2,3]) 
lam = 1
print(pdf_exp(x,lam))
print(pdf_exp2(x,lam))

# give a range of x values 
xpts=np.arange(-2,10,0.05)

# plug in to pdf_exp
y = pdf_exp(xpts, 2)

# define figure size
plt.figure(figsize=(8, 5))

plt.plot(xpts,y,'.')

plt.xlabel("x")
plt.ylabel("exponential with parameter 2 pdf ")

plt.show()





# give a range of x values 
xpts=np.arange(1e-10,10,0.05)

# define figure size
plt.figure(figsize=(12, 9))

# ggplot means graph style, 
plt.style.use('ggplot')

for l in np.arange(0.5,5,0.5):#[0.5,1,2]:#
    y = pdf_exp(xpts, l)
    plt.plot(xpts,y,'-',label = "$\lambda = %.1f$"%l)
    plt.xlabel("x")
    plt.ylabel("exponential pdf ")
    
plt.legend()
plt.title("pdf of an Exponential distribution varying parameter $\lambda$");
plt.show()





######################################################################
## An array of N random numbers in the half-open interval [0.0, 1.0) 
## can be generated using np.random.rand(N)
######################################################################
import numpy as np

np.random.seed(293423)
np.random.rand(5)

######################################################################
## when assign seed before rand(N)
## generate the same random numbers each time
######################################################################

for i in range(1,10):
    np.random.seed(2)
    print(np.random.rand(5))
    

######################################################################
## when not assign seed, generate the different 
## random numbers each time
######################################################################

for i in range(1,10):
    #np.random.seed(2) 
    print(np.random.rand(5))

arr_X = np.random.rand(10)
arr_X



