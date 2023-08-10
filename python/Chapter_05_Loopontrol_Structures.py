# Program to print natural numbers from 1 to n
# Variable decalration

n = 15

print "Enter value to n :",n

# Loop to print natural numbers
for r in range(15):
    r = r + 1
    print r,
    
    

# Program to find the value of y and print a table for values of x
# Variable declaration
import math
x = 1.0
print "------------------------"
print "     x          y       "
print "------------------------"
while x <= 3.2:
    y = 1.36 * ((1 + x + x * x * x) ** 0.5) + ((x) ** (1.0/4)) + math.exp(x)
    print "     %0.2f   " %x  + "   %0.2f"  % y
    x = x + 0.2

# Program to find factorial of given number
# Variable declaration

k = 4
kfact = 1

print "Enter an integer :",k

# Loop to generate numbers from 1 to n

for i in range(1,k + 1):
       kfact = kfact*i

print "4 factorial is",kfact    

# Program to print sum of the following series
# Variable declaration

n = 4
s = 0

print "Enter value to N :",n
# Calculation of sum
i = 1
j = 1

for i in range(1,n+1):
    term = 0
    for j in range(i+1):
        term = term + j
    s = s + term


print "Sum of the series S =",s    

# Program to compute the values of z based on x and y
# Variable declaration

x = -1.5
y = 0

# Loops to generate values of x and y to find z

while x <= 1.5:
    while y <= 3.0:
        z = 3 * x * x + 2 * y * y * y - 25.5
        print "Value of y(",x,
        print ",",y,
        print ") =",z
        y = y + 1.0
    x = x + 0.5
    y = 0

# Program to find sum of odd integers between 1 to n
# Variable declararion

n = 10

print "Enter value to N :",n

s = 0
i = 1

while (i <= n):
    s = s + i
    i = i + 2

print "Sum of odd integers =",s

# Program to generate first 50 positive integers that are divisible by 7
# Variable declaration

n = 7
print "Integers divisible by 7"

#loop to print numbers divisible by 7
for n in range(7,353,7):
    print n,
    

    

# Program to print integers from 1 to n that are not divisible by 7
# Variable declaration

n = 20
print "Enter the end value N :",n

# Loop to print numbers that are not divisible by 7

print "Integers not divisible by 7"
for k in range(1,n + 1,1):
    r = k % 7
    if (r != 0):
        print k,

# Program to print sum of digits of an integer
# Variable declaration

n = 2466

print "Enter a positive integer :",n

q = n
s = 0

while (q > 0):
    r = q % 10
    s = s + r
    q = q / 10


print "Sum of digits =",s

# Program to check whether a given number is an armstrong number or not
# Variable declaration

n = 153
q = n
s = 0

print "Enter an integer number :",n

# To check armstrong or not

while (q > 0):
    r = q % 10
    s = s + r * r * r
    q = q / 10

if (n == s):
    print "%d is an Armstrong number" % n
else:
    print "%d is not an Armstrong number" % n


        

# Program to reverse a given integer
# Variable declaration

n = 18532
q = n
rn = 0

print "Enter an integer number :",n
# Reversing

while (q > 0):
    r = q % 10
    rn = (rn * 10) + r
    q = q / 10


print "18532 is reversed as",rn   

# Program to accept an integer and print digits using words
# Variable declaration

n = 4352
q = n
rn = 0

print "Enter an integer number :",n
# converting to digits

while (q > 0):
    r = q % 10
    rn = rn * 10 + r
    q = q / 10


while (rn > 0):
    r = rn % 10
    if (r == 1):
        s = "One"
        print s,
    elif (r == 2):
        s = "Two"
        print s,
    elif (r == 3):
        s = "Three"
        print s,
    elif (r == 4):
        s = "Four"
        print s,
    elif (r == 5):
        s = "Five"
        print s,
    elif (r == 6):
        s = "Six"
        print s,
    elif (r == 7):
        s = "Seven"
        print s,
    elif (r == 8):
        s = "Eight"
        print s,
    elif (r == 9):
        s = "Nine"
        print s,
    elif (r == 0):
        s = "Zero"
        print s,
    rn = rn / 10



    

# Program to find whether a number is prime or not
# Variable declaration

n = 17

print "Enter a positive integer :",n

# prime numbers are greater than 1
if n > 1:
    for i in range(2,n):
        if (n % i) == 0:
            print(n,"is not a prime number")
            print(i,"times",n//i,"is",num)
            break
    else:
        print "%d is a prime number" % n

else:
    print "%d is not a prime number" % n

# Program to generate Fibonacci series
# Variable declaration

n = 25
n1 = 0
n2 = 1

print "Enter the final term of the series :",n

print n1,n2,


newterm = n1 + n2

# Loop to print fibonacci series
while (newterm <= n):
    print newterm,
    n1 = n2
    n2 = newterm
    newterm = n1 + n2


    

# Program to solve the sine  series
# Variable declaration

x = 0.52
n = 10
s = 0
term = x
i = 1

print "Enter x in radians :",x
print "Enter end term power (n):",n

while (i <= n):
    s = s + term
    term = (term * x * x *(-1)) / ((i + 1) * (i + 2))
    i = i + 2

print "Sum of the series = %0.6f" % s

# Program to  solve the cosine series
# Variable declaration

x = 0.52
s = 0
term = 1
i = 0
n = 10
print "Enter x in radians :",x

# Calculation of cosine series

while (i <= n):
    s = s + term
    term = (term * x * x * (-1)) / ((i + 1) * (i + 2))
    i = i + 2


print "Sum of the series = %0.6f" % s    

import math

# Program to compute the value of pie
# Variable declaration

s = 0.0
dr = 1
sign = 1
term = (1.0/dr) * sign
while (math.fabs(term) >= 1.0e-4):
    s = s + term
    dr = dr + 2
    sign = sign * -1
    term = (1.0 / dr) * sign

pie = s * 4

print "Value of pie is %.6f"%pie

# Program to evaluate the series
# Variable declaration

n = 15
s = 0.00

print "Enter value to N :",n

# Evaluation of series

for i in range(1,n + 1,1):
    s = float(s) + float(1.0 / i)
    

print "Sum of series = %0.4f" % s    
    

# Program to print multiplication table
# Variable declaration

i = 1
j = 1

# nested loop to print multiplication tables

for i in range(1,6):
    print "Multiplication table for",i
    for j in range(1,11):
        print " ",j,"x" , i ," =" , j*i
    print "Press any key to continue. . ."    

import math

# Program to convert a binary number to a decimal number
# Variable declaration

q = 1101
s = 0
k = 0

print "Enter the binary number :",q
while (q > 0):
    r = q % 10
    s = s + r * pow(2,k)
    q = q / 10
    k = k + 1


print "The decimal number is :",s

# Program to convert a decimal number to a binary number
# Variable declaration

n = 28
q = n
rbi = 0
flag = 0
k = 0

print "Enter the decimal number :",n


while (q > 0):
    r = q % 2
    if (r == 0) and (flag == 0):
        k = k + 1
    else:
        flag = 1
    rbi = rbi * 10 + r
    q = q / 2

q = rbi
bi = 0
while (q > 0):
    r = q % 10
    bi = bi * 10 + r
    q = q / 10
    i = 1
    if (q == 0):
        while (i <= k):
            bi = bi * 10
            i = i + 1


print "The binary number is ",bi            

