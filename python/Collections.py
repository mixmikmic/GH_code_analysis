x = [4,2,6,3] #Create a list with values
y = list() # Create an empty list
y = [] #Create an empty list
print(x)
print(y)

x=list()
print(x)
x.append('One') #Adds 'One' to the back of the empty list
print(x)
x.append('Two') #Adds 'Two' to the back of the list ['One']
print(x)

x.insert(0,'Half') #Inserts 'Half' at location 0. Items will shift to make roomw
print(x)

x=list()
x.extend([1,2,3]) #Unpacks the list and adds each item to the back of the list
print(x)

x=[1,7,2,5,3,5,67,32]
print(len(x))
print(x[3])
print(x[2:5])
print(x[-1])
print(x[::-1])

x=[1,7,2,5,3,5,67,32]
x.pop() #Removes the last element from a list
print(x)
x.pop(3) #Removes element at item 3 from a list
print(x)
x.remove(7) #Removes the first 7 from the list
print(x)

x.remove(20)
print(x)

y=['a','b']
x = [1,y,3]
print(x)
print(y)
y[1] = 4
print(y)

print(x)

x="Hello"
print(x,id(x))
x+=" You!"
print(x,id(x)) #x is not the same object it was
y=["Hello"]
print(y,id(y))
y+=["You!"] 
print(y,id(y)) #y is still the same object. Lists are mutable. Strings are immutable

def eggs(item,total=0):
    total+=item
    return total


def spam(elem,some_list=[]):
    some_list.append(elem)
    return some_list

print(eggs(1))
print(eggs(2))

print(spam(1))
print(spam(2))

#The for loop creates a new variable (e.g., index below)
#range(len(x)) generates values from 0 to len(x) 
x=[1,7,2,5,3,5,67,32]
for index in range(len(x)):
    print(x[index])

list(range(len(x)))

x=[1,7,2,5,3,5,67,32]
for element in x: #The for draws elements - sequentially - from the list x and uses the variable "element" to store values
    print(element)

mktcaps = {'AAPL':538.7,'GOOG':68.7,'IONS':4.6}

mktcaps['AAPL'] #Returns the value associated with the key "AAPL"

mktcaps['GS'] #Error because GS is not in mktcaps

mktcaps.get('GS') #Returns None because GS is not in mktcaps

mktcaps['GS'] = 88.65 #Adds GS to the dictionary
print(mktcaps) 

del(mktcaps['GOOG']) #Removes GOOG from mktcaps
print(mktcaps)

mktcaps.keys() #Returns all the keys

mktcaps.values() #Returns all the values

