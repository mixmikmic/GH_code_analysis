# We import numpy to get access to pi. We import numpy 
# as "np" simply to reduce the amount of typing we have to do.

import numpy as np

deg = 75.0
rad = deg * np.pi / 180.0
rad

import cmath

# Input two sides of a 3,4,5 triangle.

side1 = 3+0j
side2 = 0+4j

# Calculate the 3rd side.

side3 = side1 + side2

# Print the real and imaginary parts of the 3rd side.

print("the real part of side3 is", side3.real)
print("the imaginary part of side3 is", side3.imag)

# Calculate the length of the 3rd side using two different techniques.

print("the length of side3 is", abs(side3))
print("the length of side 3 is also", (side3.real**2 + side3.imag**2)**0.5)

# Calculate the angle of the 3rd side from the horizontal 
# using two different techniques.

print("the arg of side3 is", cmath.polar(side3)[1])
print("the arg of side3 is also", np.arctan(side3.imag / side3.real))

# Start at the origin.

a = 0+0j

# Add 8 unit vectors in a row, each rotated by 45 degrees 
# from the last, thereby producing an octagon.

for i in range(8):
    a += cmath.rect(1.0, i * np.pi / 4)
    print(a)

# Here are our two input vectors, written as Python lists:

a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]

# Calculate the dot and cross products the hard way.

dot_product = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

cross_product = [a[1]*b[2] - a[2]*b[1],
                 a[2]*b[0] - a[0]*b[2],
                 a[0]*b[1] - a[1]*b[0]]

print("the dot product is", dot_product, "the cross product is", cross_product)

# What happens when we "add" two lists?

print("a + b using lists is", a + b)

# Now do the same operations using numpy arrays, created from 
# the original lists.

a = np.array(a)
b = np.array(b)
print("the dot product is", np.dot(a, b), "the cross product is", np.cross(a, b))
print("a + b using numpy arrays is", a + b)

# Write a function called "mean" to calculate the mean
# of its argument "a".  Note that the function doesn't 
# specify what data type "a" is. For the function to work,
# "a" must be iterable (i.e., consist of one of more elements
# that can be extracted one at a time), and have
# numeric elements, else it fail at "run-time" - i.e., when 
# you try to run it - even though it passes a syntax check.
# As a consequence, this function works with lists of 
# numbers, tuples of numbers, and even complex numbers.

def mean(a):
    sum = 0.0
    for x in a:
        sum += x
    return sum / len(a)

print("the mean of (1,2,3) is", mean((1,2,3)))
print("the mean of [1,2,3] is", mean([1,2,3]))
print("the mean of (1+2j, 2+1j) is", mean((1+2j, 2+1j)))

# Calculate standard deviations using three different
# techniques. See the file stats-notes.pdf in this respository
# for details of the equations behind the program.

# First, we calculate the standard deviation using the 
# canonical formula, which requires two passes over the data.
# For large datasets, and where speed is critical, having to
# make two passes can be a problem.

def stdev0(a):
    m = mean(a)
    sum = 0.0
    for x in a:
        sum += (x - m)**2
    return (sum / (len(a) - 1))**0.5

# Next we try using the mathematically equivalent "clever" 
# simplification, requiring only one pass through the data.
# This method is actually OK if you aren't limited by 
# floating-point precision.

def stdev1(a):
    sumx = 0.0
    sumxx = 0.0
    for x in a:
        sumx += x
        sumxx += x**2
    return ((sumxx - sumx**2 / len(a)) / (len(a) - 1))**0.5

# The next function is a close-to-optimal technique to correct for
# rounding errors. It requires two passes through the data, the first
# one to find the mean.

def stdev2(a):
    m = mean(a)
    sumx = 0.0
    sumxx = 0.0
    for x in a:
        sumx += x - m
        sumxx += (x - m)**2
    return ((sumxx - sumx**2 / len(a)) / (len(a) - 1))**0.5

# Now try the above functions with an example list of three numbers.

a = [1, 2, 3]

print("stdev0 of [1,2,3] is", stdev0(a))
print("stdev2 of [1,2,3] is", stdev1(a))
print("stdev2 of [1,2,3] is", stdev2(a))

# If we add a constant to each element, the standard deviation 
# should not change.

a = [1 + 5e15, 2 + 5e15, 3 + 5e15]

# However, you find that each technique gives a different answer.

print("stdev0 of [1+ 5e15, 2 + 5e15, 3 + 5e15] is", stdev0(a))
print("stdev1 of [1+ 5e15, 2 + 5e15, 3 + 5e15] is", stdev1(a))
print("stdev2 of [1+ 5e15, 2 + 5e15, 3 + 5e15] is", stdev2(a))

# Let's try using the built-in Python function.

import statistics
print("statistics.stdev [1+ 5e15, 2 + 5e15, 3 + 5e15] is", statistics.stdev(a))

# Finally, let's use numpy. Note that numpy's standard deviation
# routine has many arguments, and you need to specify ddof=1 
# (degrees of freedom = 1) to get the expected behaviour.

b = np.array([1, 2, 3])
print("numpy.std of [1,2,3] is", np.std(b, ddof=1))

b = np.array([1.0 + 5e15, 2.0 + 5e15, 3.0 + 5e15])
print("numpy.std of [1+ 5e15, 2 + 5e15, 3 + 5e15] is", np.std(b, ddof=1))



