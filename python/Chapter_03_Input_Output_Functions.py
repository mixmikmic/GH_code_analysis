#program to add and find product of two numbers

#variable declaration
a = 5
b = 3

# calculation of sum and product 

summ = a + b
product = a * b

print a,b
print summ,product

# Program to convert degree fahrenheit to celcius 
# variable declaration

f = 105.00
print "Degree fahrenheit ? %d" % f
# calculation of degree in celcius 

c = 5.0/9.0 * (f-32)

# result 
print
print "Degree centigrade =%6.2f" % c

# To find area of a triangle
# variable declaration

a = 5
b = 4
c = 6


# calculation of area
s = float((a+b+c))/2
area = (s*(s-a)*(s-b)*(s-c)) ** 0.5

# result
print "Enter three sides : %d" % a,b,c
print 
print "Area of triangle = %0.2f Sq.units" % area

# To print ASCII value of a given character 
# Variable declaration

ch = "A"
print "Enter a character : " , ch

# Calculation of ASCII value of a character

print 
print "ASCII value of " + ch + " is" ,ord(ch)
print "Press any key to stop. . ."

# To print electricity for consumers
# Variable declaration

sno = "TMR65358"
pmr = 4305
cmr = 4410

print "Enter service number :" ,sno
print "Previous meter reading ?",pmr
print "Current meter reading ?",cmr

# Calculation of electricity charges

units = cmr - pmr
amt = units * 1.50

# Result

print
print "         Electricity Bill"
print "         ----------------"
print "Service No :",sno
print "Unit Consumed :",units
print "Electricity Charges : Rs.%0.2f" % amt

# Program to swap value of two variables
# Variable declaration

a = 15
b = 250 

print "Enter value to A :",a
print "Enter value to B :",b

# Swapping

temp = a
a = b
b = temp

print 
print "Value of A =",a
print "Value of B =",b

