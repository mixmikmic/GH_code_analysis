# Program to find the biggest of two numbers
# Variable declaration

a = 5
b = 8

print "Enter two numbers : %d" % a,b

# Calculation

big = a
if (b>big):
    big = b

print "Biggest number is ",big

# Program to find biggest of three numbers
# Variable declaration

a = 5
b = 13
c = 8

print "Enter three numbers : %d" % a,b,c

# Calculate

big = a
if (b>big):
    big = b
if (c>big):
    big = c

print "Biggest number is",big

# Program to find biggest of three numbers
# Variable declaration

a = 18
b = -5
c = 13
 
print "Enter three numbers : %d" % a,b,c

# Calculation to find biggest number

if (a>b):
    if(a>c):
        big = a
    else:
        big = c
else:
    if(b>c):
        big = b
    else:
        big = c

print "Biggest number is",big

# Program to find the value of y 
# Variable declration

x = 0.42
n = 5

print "Enter value to x and n :" , x,n 

# Calculation

if (n==1):
    y = 1 + x
elif (n==2):
    y = 1 + x / n
elif (n==3):
    y = 1 + (x ** n)
else:
    y = 1 + n * x


print "Value of y(x,n) = %0.2f" % y

# Program to find the value of y
# Variable declaration

x = 0.42
n = 5


print "Enter value to x and n :", x,n

# Calculation

# Switch case statements 
if n == 1: # case 1
    y = 1 + x
elif  n == 2: # case 2
    y = 1 + x / n
elif n == 3: # case 3
    y = 1 + (x ** n)
else: # default
    y = 1 + n * x

print "Value of y(x,n) = %0.2f" % y 

# Program to caculate the commission for sales representatives
# Variable declaration

sales = 4500

print "Sales amount ? :" , sales

# Calculation of commission

if (sales <= 500):
    comm = 0.05 * sales
elif (sales <= 2000):
    comm = 35 + 0.10 * (sales - 500)
elif (sales <= 5000):
    comm = 185 + 0.12 * (sales - 2000)
else:
    comm = 0.125 * sales


print "Commission Amount Rs.%0.2f" % comm

# Program to find roots of a quadratic equation
# Variable declaration

a = 1
b = 3
c = 2

print "Enter coefficients a, b, and c :", a,b,c

d = b * b - 4 * a * c

# Calculation of roots

if (d > 0):
    x1 = (-b + (d ** 0.5)) / (2 * a)
    x2 = (-b - (d ** 0.5)) / (2 * a)
    print "Roots are real and unequal "
    print x1,x2
    

elif(d == 0):
    x = -b / (2 * a)
    print "Roots are real and equal"
    print "%6.2f" % x

else:
    print "No Real roots, roots are complex"



    

# Program to print grade
# Variable declaration

avg_marks = 84

print "Average marks ?",avg_marks

# Calculation of grade

if (avg_marks >= 80) and (avg_marks <= 100):
    print "Honours"
elif (avg_marks >= 60) and (avg_marks <=79):
    print "First Division"
elif (avg_marks >= 50) and (avg_marks <= 59):
    print "Second Division"
else:
    print "Fail"

# Program to calculate electirc charges for domestic consumers
# Variable declaration

units = 348

print "Enter consumed units :",units

# Calculation of electric charges

if (units <= 200):
    amt = 0.5 * units
elif (units <= 400):
    amt = 100 + 0.65 * (units - 200)
elif (units <= 600):
    amt = 230 + 0.8 * (units - 400)
else:
    amt = 425 + 1.25 * (units - 600)
print 
print "Amount to be paid Rs.%0.2f" % amt

# Program to find the grade of steel samples
# Variable declaration

ts = 800
rh = 180
cc = 3

print "Enter tensile strength :",ts
print "Enter rockwell hardness :",rh
print "Enter carbon content :",cc

# Calculation of grade

if (ts >= 700):
    if (rh >= 200):
        if (cc <= 6):
            print "Grade is A"
        else:
            print "Grade is B" 
    elif (cc <= 6):
        print "Grade is C"
    else:
        print "Grade is E"      
elif (rh >= 200):
    if (cc <= 6):
            print "Grade is D"
    else:
            print "Grade is E"
elif (cc <= 6):
        print "Grade is E"
else:
        print "Grade is F"

