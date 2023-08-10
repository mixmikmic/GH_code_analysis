

for i in range(100):
    if( i%5 == 0 ):
        print(i, 'divisible by 5')
    
    if( i%7 == 0 ):
        print(i, 'divisible by 7')
        
    if( i%35 == 0):
        print('both')
    #print('do your work here')



z = 1.1
#1.print the value of z and type of z. 
print('value z = ', z,', type z = ', type(z) )

#2.define a variable   , whose value is  z^7 round at 3 decimal place.
x = round(z**7,3)

#3 print the value of  x  and type of  x
print('value x = ', x,', type x = ', type(x) )

#4 transform x to float by using the built-in function float
x = float(x)

#5 print the value of  xx  and type of  xx .
print('value x = ', x,', type x = ', type(x) )

#6 transform x to string
x = str(x)

#7 print the value of  x  and type of  x .
print('value x = ', x,', type x = ', type(x) )



#1. define variable  y  as the sum of string '123' and '456'
y =  '123' + '456'

#2. transform y into a float type
y = float(y)

#3. define variable  S  as the product of 123 and 456
s = 123*456

#4 if y>s, print y-s
if( y>s ):
    print(y-s)



