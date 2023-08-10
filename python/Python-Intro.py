## Run this cell by pressing shift + enter or the play button above.

print('Hello world')

# Numbers can be used to do simple arithmetic

1+1

# Note that only the last thing is printed out. This is because a 'statement' in computer language often returns a
# value. Because any number of computations can be done beforehand, a statement alone does not print anything unless 
# it is the last one executed.

2*2
2/.5

## Strings are just series of characters in quotes that the computer doesn't analyze one-by-one

'this_string'

# Strings are extremely simple, but can be manipulated in more complex ways as will be addressed further down

print("Hello")
print('World') 
print("She said 'Hello World'") # Both types of quotes work, the only difficulty is when you put quotes inside quotes

## Assign variables

a = 2
b = a+a  # here, b = a+a, where a is a 2 in computer memory so b = a(2)+a(2) = 4

print(b)

# Usually we name variables with a formality to be as specific to the scope as possible using underscores for space
your_name = 'fill in your name'

print(your_name)

# You can also update variables, because once it is named in a program, 
# it stays defined that way unless it is indented in special cases.
 
c = a*b
print(c)
c = your_name
print(c)

# Creating a list is just like naming any other variable, except for how you refer to the elements

my_list = [1,2,3,4,5]
my_list

# Lists can be made of any sort of collection of objects, even lists!

deep_list = [my_list,6,7,8,9,10,11,12,13,14,15]
deep_list

# Lists are also easily modified and can be modified using indexical notation

deep_list[0]  #brackets after the list, give the value at that index

# You can also subset lists using semicolons in the brackets

# The first two elements only because the value after the semicolon marks the stop point, so once python reads the 
# Element at index 2, it stops subsetting and returns what it found up to that point
deep_list[0:2]

# There are also more complex ways to index lists that are also useful for initializing ranges of numbers

deep_list[1:9:2]
# Before the first colon is the start value
# After the first colon is the end value
# The third number shows the incremental indexical jumps between values

deep_list[::-1]  # If nothing is put in the colons, then it operates on the whole list

# Lists and arrays can be created with methods as well such as range() or np.arange()
# What you call these methods on creates the list and has similar conventions to the indexing described above,
# However, they are stored only as three-values (beginning, end, increment) to save storage

short_range = range(5)
long_range = range(-5,5,2)
short_range[2:3]   # Ranges can also be subset, but the result will still be a range with different start, end, increments

## Strings

my_name = "Michael Mahoney"
print(my_name)

my_name = "Michael"   #creates memory that my_name refers to. That memory holds the string "Michael"
print(my_name)
my_name += "Mahoney"  # +=/-= is a way of abbreviating x = x + y. so (x += y) = (x = x + y) 
print(my_name)        # also shows that + joins strings

print(my_name[1])   
print(my_name[:2])
print(my_name[-3:])
print(type(my_name))

## Lists

my_first_list = [11,22,33]
print(my_first_list)

my_first_list + [44]
print(my_first_list)

my_first_list += [44]
print(my_first_list)

my_first_list.append(55)
my_first_list.append(66)

print("The length of my_list is ", len(my_first_list))
print("The first element of my_list is ", my_first_list[0])
print("The second through fourth elements of my_list are ", my_first_list[1:4]) 
## Careful, last element is NOT included in that.

## Tuples

my_first_tuple = (1, 2, 3)
print(my_first_tuple)

my_first_tuple += (4, 5, 6)
print(my_first_tuple)

print(type(my_first_tuple))

## Tuples are similar to but different than lists; first use lists before tuples, then figure out how to use tuples; we will use a tuple below

## Dictionaries: key-value pairs
## We won't spend as much time on this, but it's a good data structure to know.

my_first_dict = {"name":"Michael", "score1":100}
print(my_first_dict)
my_first_dict["score2"] = 90
print(my_first_dict)

print(my_first_dict["name"])

print(my_first_dict.keys())
print(my_first_dict.values())

print(type(my_first_dict))

## More complex printing

my_variable = 10

x = "The value of my variable is %s"
print(x, my_variable)

x = "The value of my variable is %s" % my_variable
print(x)

print("The value of my variable is %s" % my_variable)

print(type(my_variable))
print(type(x))

## More complex printing, cont.

first = "Michael"
last = "Mahoney"
full1 = first + " " + last
print(full1)
full2 = "%s %s" % (first, last)
print(full2)

print(type((first, last)))
## Note that that is a tuple.
print(type(full1))
print(type(full2))

## More complex printing, cont.

pi_to_8decimals = 3.14159265
print("pi is approximately equal to %.3f" % pi_to_8decimals)
print("pi is approximately equal to %8.3f" % pi_to_8decimals)
print("pi is approximately equal to %8.5f" % pi_to_8decimals)

## The following will result in an error since the variable isnt defined.

pi

## The following will result in an error since the math library hasnt been imported yet

math.pi

## Import the math library.  This is one of many libraries.

import math

## The following will result in an error since the variable still isnt defined.

pi

## But it is defined in the math library, so we can access it this way.

math.pi

## More complex printing, cont.

print('pi is roughly: %.30f' % math.pi)
print(' e is roughly: %.30f' % math.e) 

print('pi is roughly: %.10f' % math.pi)
print(' e is roughly: %.10f' % math.e) 

## For loops.  Very important to do stuff.  Be careful with indentations.

for counter in [1,2,3,4]:
    print(counter)
    print("Still in the first loop")
    
print("Out of the first loop")

for x in range(0, 9,2):
    print("In second loop: %d" % x)
    print("In second loop: %d" % (x)) 

    print("In second loop: %d %d" % (x, x))
    
    
print("Finished and out of second loop")

## Careful: different ways to sum numbers give different answers.
## This wont be too much of an issue for us, but always check what every step of your computation does.

values = [ 0.1 ] * 10

print('Input values:', values)

print('sum()       : {:.20f}'.format(sum(values)))

ss = 0.0
for ii in values:
    ss += ii
    
print('for-loop    : %.20f' % ss)

print('for-loop    : {:.20f}'.format(ss))
    
print('math.fsum() : {:.20f}'.format(math.fsum(values)))

## Functions.  A good way to reuse code.

def get_circumfrence(r):
    c = 2*math.pi*r
    return(c)

def get_area(r):
    """
    This is a function to compute the area of a circle of radius r
    """
    a = math.pi*r*r
    return(a)

r1 = 10
c1 = get_circumfrence(r1)
a1 = get_area(r1)
r2 = 20
c2 = get_circumfrence(r2)
a2 = get_area(r2)
print("For a radius of %.3f, the circumfrence is %.3f and the area is %.3f" % (r1,c1,a1))
print("For a radius of %.3f, the circumfrence is %.3f and the area is %.3f" % (r2,c2,a2))

def get_bounding_box_area(r):
    bba = 2**2 * r**2
    return(bba)

print("")
for ii in range(0,21):
    print("Radius:\t%10.3f\tCircumfrence:\t%10.3f\tArea:\t%10.3f\tBoundingBoxArea\t:%10.3f" %(ii,get_circumfrence(ii),get_area(ii),get_bounding_box_area(ii)) ) 

## Arrays.  If you go to places like stackoverflow (a great resource), you may find someting like the following.
## But it may not work, if you don't have the right things loaded.

np.array([1,2,3,4,5])

## Numpy
## Here import numpy.  The "numpy as np" isnt strictly necessary, but it is pretty common.
import numpy as np

## Arrays.   

np.array([1,2,3,4,5])

np.array([[1,2],[3,4],[5,6]])

np.zeros((3,4))

np.ones((3,4))

np.linspace(0,math.pi,num=101)

## A random number between 0 and 1
np.random.random()

## An array of random numbers between 0 and 1
np.random.random((2,3))

## Simple statistics to compute

dataset_rn = np.random.random((2,3))
dataset_rn
print( np.max(dataset_rn) )
print( np.max(dataset_rn, axis=0) )
print( np.max(dataset_rn, axis=1) )
print( np.min(dataset_rn) )
print( np.mean(dataset_rn) )
print( np.median(dataset_rn) )
print( np.std(dataset_rn) )
print( np.sum(dataset_rn) )

## Reshape and manipulate an array of numbers

dataset_rn_reshaped1 = np.reshape(dataset_rn, (3,2))
dataset_rn_reshaped1

dataset_rn_reshaped2 = np.reshape(dataset_rn, (1,6))
dataset_rn_reshaped2

dataset_rn_reshaped3 = np.reshape(dataset_rn, (6,1))
dataset_rn_reshaped3

dataset_rn_reshaped4 = np.reshape(dataset_rn, (6))
dataset_rn_reshaped4

## Make it slightly bigger
dataset_rn = np.random.random((5,10))
dataset_rn

## Select the first row which means the second row since counting starts at zero.
dataset_rn[1]

## Remember indexing starts at 0.
print(dataset_rn[0][0])

## The following are the same.
print(dataset_rn[1][2])
print(dataset_rn[1,2])

## But the following are different.
print(dataset_rn[0:2][0:3])
print("")
print(dataset_rn[0:2,0:3])

## How to access the 0th column or row.
print(dataset_rn[0])
print("")
print(dataset_rn[0,])
print("")
print(dataset_rn[:,0])

## Plotting with Math Plot Lib: we have to import another library

import matplotlib.pyplot as plt

x1 = [-2,-1,0,1,2]
x2 = [4,1,0,1,4]

## Do a first plot.

fig = plt.figure()
axes = fig.add_subplot(111)
axes.plot(x1,x2)

## QUESTION: How do I get the X axis and Y axis on this?

x1_highres = np.linspace(-2,2,40)
x2_highres = x1_highres**2

fig = plt.figure()
axes = fig.add_subplot(111)
axes.plot(x1,x2)
axes.plot(x1_highres,x2_highres)

## QUESTION: What does the following do?  I seem not to need it for what Im doing.  Is it related to the above issue?
plt.show()

## Lots of arguments can be given to plotting functions.  Its worth knowing how to use them.

fig = plt.figure()
axes = fig.add_subplot(111)
##
## Comment and uncomment each of the following in turn to see how the plot changes
##
#axes.plot(x1,x2)
#axes.plot(x1_highres,x2_highres)

#axes.plot(x1_highres,x2_highres,color="blue",linestyle="solid")
#axes.plot(x1_highres,x2_highres,color="green",linestyle="dashed")
#axes.plot(x1_highres,x2_highres,color="green",linestyle="dashdot")
#axes.plot(x1_highres,x2_highres,color="red",linestyle="dotted")

#axes.plot(x1_highres,x2_highres,color="red",linestyle="dashed",linewidth=3)
#axes.plot(x1_highres,x2_highres,color="red",linestyle="dashed",linewidth=1,marker='o')
#axes.plot(x1_highres,x2_highres,color="red",linestyle="dashed",linewidth=1,marker=',')
#axes.plot(x1_highres,x2_highres,color="red",linestyle="dashed",linewidth=1,marker='v')
#axes.plot(x1_highres,x2_highres,color="red",linestyle="dashed",linewidth=1,marker='^')
#axes.plot(x1_highres,x2_highres,color="red",linestyle="dashed",linewidth=1,marker='s')
#axes.plot(x1_highres,x2_highres,color="red",linestyle="dashed",linewidth=1,marker='p')
#axes.plot(x1_highres,x2_highres,color="red",linestyle="dashed",linewidth=1,marker='*')

#axes.plot(x1_highres,x2_highres,color="red",linestyle="dashed",linewidth=3,marker='o',markerfacecolor='blue')
axes.plot(x1_highres,x2_highres,color="red",linestyle="dashed",linewidth=3,marker='o',markerfacecolor='blue',markersize=5)

fig = plt.figure()
#fig = plt.figure(figsize=(12,8))
axes = fig.add_subplot(111)
axes.plot(x1,x2)
axes.plot(x1_highres,x2_highres)
axes.set_title("$y=x^2$")  
## The thing above between the dollar signs is latex, which is nice for equations, but dont worry if you dont know it.
## You can just put text in the quotes if you like.
#axes.grid()
axes.set_xlabel("xvalue")
axes.set_ylabel("yvalue")
#plt.show()

## Plot a sinusoid and exponential

radians_to_degrees = 180.0/math.pi

x1vals = np.arange(0, 6*math.pi, 0.2)
x2vals = np.cos(x1vals)
x3vals = np.exp(-x1vals/(2*math.pi))

if False:
    print(x1vals)
    print(x2vals)
    print(x3vals)

#fig = plt.figure()
#axes = fig.add_subplot(111)
#axes.plot(x1vals,x2vals)
#axes.plot(x1vals,x3vals)

plt.scatter(x1vals, x2vals)
plt.scatter(x1vals, x3vals,color="red")

## QUESTION: Why an I getting the following warning?

## Plot a noisy exponential and a noisy sinusoid

def noisifyAbs(yvals_in):
    gamma = 0.5
    noise = gamma*np.max(yvals_in)*(np.random.random(yvals_in.shape)-0.5)
    yvals_out = yvals_in + noise
    return(yvals_out)

def noisifyRel(yvals_in):
    gamma = 0.5
    noise = gamma*np.max(yvals_in)*(np.random.random(yvals_in.shape)-0.5)
    yvals_out = []
    for ii in range(0, len(noise)):
        yvals_out.append(yvals_in[ii]*(1+noise[ii]))
    return(yvals_out)

radians_to_degrees = 180.0/math.pi

x1vals = np.arange(0, 6*math.pi, 0.5)
x2_clean = np.cos(x1vals)
x3_clean = np.exp(-x1vals/(2*math.pi))

x2_noisyAbs = noisifyAbs( x2_clean )
x3_noisyAbs = noisifyAbs( x3_clean )
x2_noisyRel = noisifyRel( x2_clean )
x3_noisyRel = noisifyRel( x3_clean )

#print(x2_noisyAbs)
#print(x3_noisyAbs)
#print(x2_noisyRel)
#print(x3_noisyRel) 

fig = plt.figure()
axes = fig.add_subplot(111)
axes.plot(   x1vals,    x2_clean, color="black",linestyle="solid",linewidth=1)
axes.plot(   x1vals,    x3_clean, color="black",linestyle="solid",linewidth=1)
plt.scatter( x1vals, x2_noisyAbs, color="red"     )
plt.scatter( x1vals, x3_noisyAbs, color="green"   )
plt.scatter( x1vals, x2_noisyRel, color="blue"    )
plt.scatter( x1vals, x3_noisyRel, color="magenta" )

## Take a closer look at the properties of the noise.  Which noise model is more reasonable?

fig = plt.figure()
axes = fig.add_subplot(111)
#axes.plot(   x1vals,    x2_clean, color="black",linestyle="solid",linewidth=1)
axes.plot(   x1vals,    x3_clean, color="black",linestyle="solid",linewidth=1)
#plt.scatter( x1vals, x2_noisyAbs, color="red"     )
plt.scatter( x1vals, x3_noisyAbs, color="green"   )
#plt.scatter( x1vals, x2_noisyRel, color="blue"    )
plt.scatter( x1vals, x3_noisyRel, color="magenta" )







## Lets look at a matrix of noise.

data_noise = np.random.random((100,100))
data_noise
print(type(data_noise))
print(type(np.mat(data_noise)))

tmp1 = np.mat(data_noise)
gram = np.dot(tmp1,tmp1.T)

plt.imshow(data_noise)
plt.colorbar()
plt.show()

plt.imshow(data_noise,cmap=plt.cm.gray)
plt.colorbar()
plt.show()

## What does the following plot tell us?

plt.imshow(gram)
plt.colorbar()
plt.show()



trip = []
def transform(m,n):
    for i in range(n):
        m = 1 - (1/m)
        trip.append(m)
    return m  

transform(.5,8)

import numpy as np
plots.plot(np.arange(8),trip)





