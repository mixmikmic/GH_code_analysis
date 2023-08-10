print(abs(-1))

# all values true
l = [1, 3, 4, 5]
print(all(l))

# all values false
l = [0, False]
print(all(l))

# one false value
l = [1, 3, 4, 0]
print(all(l))

# one true value
l = [0, False, 5]
print(all(l))

# empty iterable
l = []
print(all(l))

l = [1, 3, 4, 0]
print(any(l))

l = [0, False]
print(any(l))

l = [0, False, 5]
print(any(l))

l = []
print(any(l))

normalText = 'Python'
print(ascii(normalText))

otherText = 'Pythön'
print(ascii(otherText))

print('Pyth\xf6n')

print(bin(2))
print(bin(11))

print(bool(0))
print(bool(1))
print(bool(-1))
print(bool(123))
print(bool(None))

string = "Python is interesting."

# string with encoding 'utf-8'
arr = bytearray(string, 'utf-8')
print(arr)

my_val = [1, 2, 3]

# string with encoding 'utf-8'
val_arr = bytearray(str(my_val), 'utf-8')
print(val_arr)

from keyword import kwlist
print(kwlist)

