# Integers , Strings
#creating a varibale
variable = 1

print(type(variable))      # data type of the variable is printed

variable = 'This is a string'
print(variable)
print(type(variable))

#for concatenation of strings we can use the addition sign i.e. '+'
print ('This is' + 'a new string')

#printing the same element for a numbre of times you can use '*'
print('-'*70)

string = 'THis Is a StriNG'

# some built-in functions of strings 
print(string.__len__())

print(string.lower())

print(string.lower().split('s'))  # lower() is used to convert the string into small alphabets  and split() is used to split the string about any token say 's' that is encountered in the string

# Lists

# Lists can have elements of different datatypes 
l = [1,'se',2.3,[1,2,'89'],8.9]

#printing the list
print(l)

# to print the last element of list you can use -1. '-1' denotes the index of last element
print(l[-1])

# to print the element of list inside the list say '89' we can do it like
print(l[3][2])

# printing the ASCII value of any alphabet
print(ord('a'))

# similarly printing the char value of ASCII number
print(chr(100))

print('*'*100)


#let nums be list containing lists

nums = [[2,20],[4,5],[1.3],[0,-1]]
print(nums)


print(sorted(nums)) # default based on 0th index


# Dictionaries
d = {
    2:'abc',
    'p': 0.1,
    'q': [0,{0:20, 'k':120}]
}

# you can insert any kind of datatype fror lists and dictionaries

#printing the dictionary
print(d)

# changing the default value of a variable
print(d.setdefault('my','ababab'))

# changing value
d['aaa'] = 132

#printing the dictionary values
print (d.values())

#printing the dictionary keys
print(d.keys())

# Tuples
t = (1,2,3)

#prinitng the tuple
print(t)

# Sets
p = set([1,4,3,2,2,3,4,5,5,2,5,3,54,2,54,23,5])
q = set([-1,0,4,3,2,3,2,99])

print(p)
print(q)

#Else-If Condition

x = 100
if x< 5 and x>20:
    print("HH")
elif x>=12 and x<20:
    print("GG")
else:
    print("PP")

# Loops
for ix in range(len(l)):
    print (ix, l[ix])

for ix in l:
    print (ix)
    
print(';)'*50)

j = 0
while j<10:
    print(j)
    j+=1

# FizzBuzz Challenge 
'''For numbers between 0 to 30 that are divisible by 3 and 5 print "Fizzbuzz" elseif it is divisible by 3 print "Fizz" else if divided by 5 print "Buzz" and for left number print the number itself '''

l = range(1,30)

for ix in l:
    if ix % 3 == 0:
        if ix % 5 ==0:
            print("FizzBuzz")
        else:
            print("Fizz")
    else:
        if ix % 5 ==0:
            print("Buzz")
        else:
            print(ix)

# defining functions
def avg(*args,**kwargs):
    print (args)
    print(kwargs)

print(avg(2.02, 45, a=10, b=45, pq = -3))

# importing libraries
from math import sqrt as sq  # square root function in aliased as sq
print (sq(5))

# Creating classes

class MyClass:
    
    x= 100
    def __init__(self, x=1, t=10):  #constructor
        self.x = x
        self.t = t
        
    def new_a(self, a =90):
        self.a = a
        
    def squareself(self):
        self.x = x
        self.t = t
        self.a = a
        
    def divide(self,p):
        self.squareself()
        return p/(self.x)+ p/(self.t) + p/(self.a)

# creating new object
z = MyClass()

#class object
print(z)

#accessing the variable of class through object
print(z.x)

print(z.t)

#calling function of class
z.new_a()







