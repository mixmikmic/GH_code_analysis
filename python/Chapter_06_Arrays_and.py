# Program to find the sum of n numbers using array
# Variable declaration

n = 5
x = [36,45,52,44,62]

print "How many integers ?",n
print "Enter the 1th value :",x[0]
print "Enter the 2th value :",x[1]
print "Enter the 3th value :",x[2]
print "Enter the 4th value :",x[3]
print "Enter the 5th value :",x[4]

summ = 0 

# Calculation of sum of numbers
for i in x:
    summ = summ + i
    

print "Sum of all integers =",summ
    

# Program to find the biggest of n numbers
# Variable declaration

n = 5
x = [25,-228,0,185,36]

print "How many numbers ?",n
print "Enter all those numbers"
for a in x:
    print a,
print     

big = x[0]


for i in x:
    if (i > big):
        big = i


print "%d is the biggest number" % big        

# Program to find the arithmetic mean, variance and standard deviation
# Variable declaration

n = 6
x = [3.1,3.8,3.6,4.0,3.4,3.8]
summ = 0
vsum = 0

print "How many values ?",n
print "Enter all values in the list "
for a in x:
    print a,
print     

# Loop to find sum of all values

for i in x:
    summ = summ + i

xbar = summ / n

# Loop to find the numerator vsum to find variance

for i in x:
    vsum = vsum + (i - xbar) * (i - xbar)


sigmax = vsum / n
sd = sigmax ** 0.5

print "Arithmetic mean = %0.3f" % xbar
print "Variance = %0.3f " % sigmax
print "Standard deviation = %0.3f" % sd

# Program to calculate mean of marks and print list of marks greater than mean
# Variable declaration

n = 5
x = [58,63,68,54,48]
summ = 0 # Used summ instead of sum since it was a inbuilt function
i = 0

print "How many students ?",n
print "Enter all the marks "
for a in x:
    print a,
print     

for i in x:
    summ = summ + i

mean = float(summ) / n

print "Mean = %0.2f" % mean
print "Marks greater than mean :",

i = 0
for i in x:
    if (i > mean):
        print i,

        

# Program to find sum of all positive and negative numbers and to find out which is larger in magnitude
# Variable declaration
import math

n = 6
x = [8,-12,-16,12,-9,5]
psum = 0
nsum = 0

print "How many values ?",n
print "Enter all values in the list"
for i in x:
    print i,
print     

# Loop to calculate sum of positive and negative values

for i in x:
    if i > 0:
        psum = psum + i
    else:
        nsum = nsum + i

print "Sum of positive values = %0.2f" % psum
print "Sum of negative values = %0.2f" % nsum

if (psum > abs(nsum)):
    print "Positive sum is greater in magnitude"
else:
    print "Negative sum is greater in magnitude"

diff = abs(psum) - abs(nsum)
print "Difference in magnitude = %0.2f" % abs(diff)

# Program to sort n numbers in ascending order
# Variable declaration

n = 4
x = [32,-10,20,5]
i = 0


print "How many numbers ?",n
print "Enter the list of 4 numbers"
for a in x:
    print a,
print

# Loop to arrange the numbers in ascending order

while i < n-1:
    j = i + 1
    while j < n:
        if x[i] > x[j]:
            temp = x[i]
            x[i] = x[j]
            x[j] = temp
        j = j + 1
    i = i + 1

print "Numbers in ascending order "

for a in x:
    print a,

# Program to search the key value and to print it if the search is successful
# Variable declaration

n = 6
x = [6,-2,8,3,13,10]
s = 3

print "How many values in the list ?",n
print "Enter all values in the list"
for i in x:
    print i,
print     
print "Enter the key value to be searched :",s

# loop to search key value in the list

for i in range(n):
    if s == x[i]:
        print s," is available in",i+1,"th location"

# Program to sort n numbers using bubble sort and find number of exchanges and passes
# Variable declaration

n = 4
x = [6,-2,8,3]
exchng = 0

print "How many numbers?",n
print "Enter all the numbers in the list"
for i in x:
    print i,

print

for i in range(0,n-1):
    for j in range(0,n-i-1):
        if x[j] > x[j+1]:
            temp = x[j]
            x[j] = x[j+1]
            x[j+1] = temp
            exchng = exchng + 1
            

print "The sorted list is"
for i in x:
    print i,

print     

print "Sorted in",n-1,"passes and",exchng,"exchanges"

# Program to add two matrices
# Variable declaration

a = [[2,-2],
     [0,4]]
b = [[6,2],
     [4,-5]]

c = [[0,0],
     [0,0]]

m = 2
n = 2

print "How many rows and columns ?",m,n
print "Enter A matrix"
for i in range(m):
    for j in range(n):
        print a[i][j],
    print 

print "Enter B matrix"
for i in range(m):
    for j in range(n):
        print b[i][j],
    print 

# Loop to add two matrices

for i in range(m):
    for j in range(n):
        c[i][j] = a[i][j] + b[i][j]


print "Resultant matrix is"
for i in range(m):
    for j in range(n):
        print c[i][j],
    print 

# Program to multiply two matrices
# Variable declaration

m = 2
n = 2
l = 2
a = [[2,-2],
     [0,4]]
b = [[6,2],
     [4,-5]]
c = [[0,0],
     [0,0]]

print "Enter order of A matrix :",m,n
print "Enter A matrix"
for i in range(m):
    for j in range(n):
        print a[i][j],
    print 
    
print "Enter order of B matrix :",m,n
print "Enter B matrix"
for i in range(m):
    for j in range(n):
        print b[i][j],
    print 

# Loop to multiply two matrices
# iterate through rowa of A
for i in range(m):
    # iterate through columns of B
    for j in range(l):
        c[i][j] = 0
        # iterate through rows of B
        for k in range(n):
            c[i][j] = c[i][j] + a[i][k] * b[k][j]


print "Resultant matrix is"

for i in range(m):
    for j in range(n):
        print c[i][j],
    print 
    

# Program to find and print the transpose of the matrix
# Variable declaration

m = 2
n = 3
a = [[-3,6,0],
     [3,2,8]]
at = [[0,0],
      [0,0],
      [0,0]]

print "Enter order of the matrix :",m,n
print "Enter the matrix values"
for i in range(m):
    for j in range(n):
        print a[i][j],
    print 

# Loop to calculate transpose

for i in range(m):
    for j in range(n):
        at[j][i] = a[i][j]

print "The transposed matrix is "
for i in range(n):
    for j in range(m):
        print at[i][j],
    print 
    

# Program to check whether a given matrix is symmetric or not
# Variable declaration

m = 3
a = [[5,3,8],
     [3,1,-7],
     [8,-7,4]]

print "Enter order of the square matrix :",m
for i in range(m):
    for j in range(n):
        print a[i][j],
    print 

# Loop to check whether symmetric or not

for i in range(m):
    flag = 0
    for j in range(m):
        flag = 0
        if a[i][j] == a[j][i]:
            continue
        else:
            flag = 1

if flag == 0:
    print "The given matrix is a symmetric matrix"
else:
    print "The given matrix is not a symmetric matrix"

# Program to find the trace of a given square matrix
# Variable dclaration

m = 3
a = [[3,2,-1],
     [4,1,8],
     [6,4,2]]
summ = 0

print "Enter order of the square matrix :",m
print "Enter the matrix"
for i in range(m):
    for j in range(m):
        print a[i][j],    
    print 

# Loop to find trace
for i in range(m):
    summ = summ + a[i][i]

print "Trace of the matrix =",summ    

