#variable
x = 1
x5 = 10

#5x = 10 variable name is wrong

# Big/small letter different variables
X = 100
print('x =', x,', X =', X)


# reassign values
X = x5
print('X = ',X)

# 2 to the power 3
p = 2**3
print(p)

# original x is 'int'
# now x is 'str' 
# Okay to assign different types of values to the same variable
x = '123'
print(x, type(x))

# string -> int
# only works for 'int' looking strings
y = int(x)
print('value = ', y,', type = ', type(y))

#print(int('abd'))  error
#print(int('12.'))  error
#print(float('12.'))

intss = 1

if(intss>2):
    print('if run!')

print('after if',intss)
print('done',intss)

intss = 1

if(intss>2):
    print('if run!')
    print('inside if',intss)

print('done',intss)



#float vs int
x = 100.111
print('value = ', x,', type = ', type(x))

x = 100.
print('value = ', x,', type = ', type(x))

x = float(100)
print('value = ', x,', type = ', type(x))

x = 100
print('value = ', x,', type = ', type(x))

x = 1e2
print('value = ', x,', type = ', type(x))

x = 100//1
print('value = ', x,', type = ', type(x))

x = 100/1
print('value = ', x,', type = ', type(x))

#string summation
'123'+'456'

def type_print(x):
    print('value = ', x, ', type = ', type(x))

listx = [100.111, 100., float(100), 100, 1e2, 100//1, 100/1]
for x in listx:
    type_print(x)

#python 3.x auto float
2*3/4



