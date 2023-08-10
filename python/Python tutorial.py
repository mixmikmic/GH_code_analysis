print("Hello World!")

#Single line comment

#Below is a multi line comment
'''Multi line comment
with three quotes'''

# integers
variableInteger = 1

# floats
variableFloat = 1.1

# simple addition
print(variableInteger + variableFloat)

# strings
variableString = 'This is a string variable'

print(variableString)

# string indexing
print(variableString[0])
print(variableString[1:])

# String functions
print(variableString.startswith('T'))

# lists
_list1 = [1,2,3,1]
_list1.append(4)
print(_list1)

# sets
_set1 = set([1,2,3,1])
print(_set1)

# dictionaries
_dict = {1: 'First', 2: 'second', 3: 'third'}
print(_dict[2])

if 1 in _dict:
    print('Value Detected')
    
if 'third' in _dict:
    print('Detected 4')
elif 0 in _dict:
    print('Detected 2')
else:
    print('Values not detected')

for x in range (10): #0-9
    print (x)

for key, value in _dict.items():
    print (key, value)

def my_add_function(a,b):
    return a+b

x = my_add_function(1,1)
print(x)



