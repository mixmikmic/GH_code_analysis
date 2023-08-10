import numpy as np #np for short
#use built-in help & dir
#dir(range)
#help(range.count)

# use lambda to define function quickly
# this is saying g maps x to x**2

g = lambda x: x**2
print(g(2))

# array iteration
b = np.arange(6).reshape(3,2)
print(b)

for r in b:
    print('iteration')
    print(r)

for (a,b) in b:
    print('iteration')
    print(a)
    print(b)

# array iteration
b = np.arange(60).reshape(3,2,10)
print(b,'\n')

for r in b:
    print('iteration')
    print(r)

# create arrays:
x1 = np.ones((2,3), dtype=float)
x2 = np.ones((2,3), dtype=int)
x3 = np.ones((2,3))

# a list of arrays
XX = [x1,x2,x3]

# '\n' means an empty line
for x in XX:
    print(x,'\n')

# Other ways to create arrays
xarange = np.arange(1, 7, 2, dtype=int)
print(xarange)

xarange = np.arange(0, 30, 5, dtype=int) # does not have 30 in it
print(xarange)

xarange = np.arange(0, 31, 5, dtype=int) # does  have 30 in it
print(xarange)

ones_r = np.ones((3,)) #1-d array: r-vector
ones_c = np.ones((3,1)) #3 by 1 matrix

print(ones_r)
print(ones_c)

# Transpose .T
# for 1-d array, doesn't work
ones_r_transpose = ones_r.T
print(ones_r_transpose, ones_r_transpose.shape)

# Transpose .T
# still 2-d: 1 by 3
ones_c_transpose = ones_c.T
print(ones_c_transpose,ones_c_transpose.shape)


#.flatten()  1-d array
ones_c_transpose_to_vector = ones_c_transpose.flatten()
print(ones_c_transpose_to_vector,ones_c_transpose_to_vector.shape)


ones_list = ones_r.tolist()
print(ones_list)

# a list does not have shape
# print(ones_list.shape) # will not run

list123 = [1,2,3]
arr123 = np.array(list123)
print(arr123.shape)


#X = np.linspace(0,4*np.pi,1000)
X = np.linspace(0,1,5)
help(np.linspace)

# array operations
a = np.array([1.1,2.4,4.6])

# an array [0 1 2]
b = np.arange(3)
print(a,b)
print(a/b) # b[0] is zero will receive warning message
print(a*b) # entry-wise product
print(a**b) # gives 1.1**0, 2.4**1, 4.6**2

# min, max, sum
a = np.array([1,4,3,1,4])

# either do np.sum() or a.sum()
print('sum',a.sum(),np.sum(a))
print('max',a.max(),np.max(a))
print('min',a.min(),np.min(a))
print('location of max',a.argmax()) #the first max shows up
print('location of min',a.argmin()) #the first min shows up


a = np.array([1,4,3,1,4])
print('before sort', a)

# a.sort() changes a into sorted order
a.sort()

print('after sort',a)


# comparison
a = np.array([-1,4,-3,1,5,4])
print(a, a>0)

# returns values in the array a, where they are greater than 0
print(a[a>0],'values in array a where are bigger than 0')


print(a)

# returns index of values in the array a, where the corresponding values are greater than 0
print(np.where(a > 0)[0], 'index of values in array a, who  are bigger than 0' )

# statistics
# percentage of array a >3
np.sum(a>3)/len(a)
a = np.arange(10).reshape(2,5)
print(a)
print(np.median(a))#treat a as one array
print(np.median(a,axis = 1)) # calculate median of each row
print(np.median(a,axis = 0)) # calculate median of each column
print(np.mean(a,axis = 1))# calculate empirical mean of each column 

# statistics
# variance
def emp_var(arr1):
    return np.sum((arr1-np.mean(arr1))**2) / (len(arr1)-1) #empirical variance

arr1 = np.array([1,2,3])
print(emp_var(arr1))
print(np.var(arr1))

# An array of random numbers in the half-open interval [0.0, 1.0) can be generated:

seed = 123
np.random.seed(123)
print('seed = ', seed , np.random.rand(5))

np.random.seed(293423)
print('seed = ', 293423,  np.random.rand(5))

np.random.seed(293423)
print('seed = ', 293423,  np.random.rand(5))


print('no seed',np.random.rand(5 ))
print('no seed',np.random.rand(5 ))
print('no seed',np.random.rand(5 ))

# when assign seed, generate the same random numbers each time
for i in range(1,6):
    np.random.seed(2)
    print(np.random.rand(3))

# when not assign seed, generate the different random numbers each time

for i in range(1,6):
    #np.random.seed(2) 
    print(np.random.rand(3))

np.random.rand(2,3)

import numpy as np
print('pi: ',np.pi)
print('sin(pi): ',np.sin(np.pi))
print('euler\'s number:', np.e)
print('natural log (e) :',np.log(np.e))
print('exponential',np.exp(1))



# need to import this, in order to use it
import matplotlib.pyplot as plt
import numpy as np

x = [88, 48, 60, 51, 57, 85, 69, 75, 97, 72, 71, 79, 65, 63, 73]


# very first histogram, 
plt.hist(x,bins = 6) # divide [minx, maxx] into 6 bins evenly

# must write plt.show() for your plot to show, otherwise there will be no plot
plt.show()


x = [88, 48, 60, 51, 57, 85, 69, 75, 97, 72, 71, 79, 65, 63, 73]


# catch information of histogram, 
#plt.hist returns a triple:
#1)  (n_d below): the count in each bin
#2) (bins_d below): the bins interval
#3) (dummy): a python structure that we don't need for now
n_d, bins_d, dummy = plt.hist(x,bins = 6,edgecolor='black', linewidth=1.2)

plt.show()
print(n_d,)
print(bins_d)

# nicer histogram with more information
# shows the bins we want to by adding: bins = 6,range = (40,100) to hist()

x = [88, 48, 60, 51, 57, 85, 69, 75, 97, 72, 71, 79, 65, 63, 73]


#plt.hist returns a triple:
#1) the first (n_d below): the count in each bin
#2) the second (bins_d below): the bins interval
#3) the third(dummy): a python structure that we don't need for now


plt.figure(figsize = (20,10)) # adjust figure size
n_d, bins_d, dummy = plt.hist(x,bins = 6, range = (40,100),
                              edgecolor='black', linewidth=1.2)
#plt.plot(bins_d,np.zeros(len(bins_d)),'y*')

plt.show()

print(n_d,)
print(bins_d)



