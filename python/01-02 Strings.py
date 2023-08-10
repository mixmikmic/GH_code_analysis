var = 'Hello World'
print("Contents of var: ", var)
print("Type of var: ", type(var))

print('C:\name\of\dir')  # even using triple quotes won't save you!

print(r'C:\name\of\dir')

var1 = 'Hello'  # String 1
var2 = 'World'  # String 2
var3 = var1 + var2  # Concatenate two string as String 3
print(var3)

var1 = 'Hello' 'World'
print(var1)

var1 = 'Hello'
var2 = 1
print(var1+var2)

var1 = 'Python'
len(var1)

var1[0]

var1[5]

var1[0] = 'J'

var1[-6]

var1[-1]

var1[0:3]

var1[:-3]

var1[:2]+var1[-4:]

text = '%s World. %s %d %d %f'
print(text)

text %('Hello', 'Check', 1, 2, 3)

text = '{} World. {} {} {} {}'
# you can also do
# text = '{0} World. {1} {2} {3} {4}'
print(text)

text.format('Hello', 'Check', 1, 2, 3)

text = '{} World. {} {} {} {:.2f}'
# you can also do
#text = '{val1} World. {val2} {val3} {val4} {val5:.2f}'
print(text)

text.format('Hello', 'Check', 1, 2, 3)
# if you uncomment the previous cell, then use this:
#text.format(val1='Hello', val2='Check', val3=1, val4=2, val5=3)

var1 = 'python'
var1.capitalize()

var1.center(10, '!')

var1.count('t', 0, len(var1))

var1 = "Hello World"
var1.endswith("World")

var2 = "This is a test string"
var2.find("is")

var3 = 'Welcome2015'
var3.isalnum()

var1 = ''
var2 = ('p', 'y', 't', 'h', 'o', 'n')
var1.join(var2)

var = '.......python'
var.lstrip('.')

var = 'python'
max(var)  # This is very helpful when used with integers

var = 'This is Python'
var.replace('is', 'was', 1)

var = 'Python'
var.rjust(10,'$')

var = 'This is Python'
var.split(' ')

