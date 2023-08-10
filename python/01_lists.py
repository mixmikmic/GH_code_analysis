l1 = list() 
l2 = []

print l1
print l2

print len(l1)
print len(l2)

l3 = [1, 2, 3]
print l3
print len(l3)

l1.append(1)
print l1
l1.append(10)
print l1

l2.append(100)
print l2 

print l1
print l2
print l1 + l2

l1.extend(l2)
print l1
print l2

l2 = [0, 1, 2, 3, 4, 5, 6]

l2.pop(1)
print l2

l2.pop()
print l2

print l1 

l3 = [-2, -1, 0, 1, 2]
print l3
print len(l3)

print [el for el in l3]

# Return the positive elements 
print [el for el in l3 if el > 0]

# Return the negative elements 
print [el for el in l3 if el < 0]

# Multiply the elements by two
print [el * 2 for el in l3]

# Multiply filtered elements by two
print [el * 2 for el in l3 if el <= 1]

def is_positive(el): 
    return el > 0

print l3
print filter(is_positive, l3)

# Return the positive elements
print filter(lambda el: el > 0, l3)

# Return the non-positive elements
print filter(lambda el: el <= 0, l3)

# Return elements outside of a range
print filter(lambda el: el < -1 or el > 1, l3) 

# Return the elements found within a range (note the mathematical notation)
print filter(lambda el: -1 <= el <= 1, l3)

print [abs(el) for el in l3]

print map(abs, l3)

def add_one(item): 
    return item + 1

print map(add_one, l3)

print map(lambda el: el * 2, filter(lambda el: el <= 1, l3))

print 'Integer array:', map(int, l3)
print '  Float array:', map(float, l3)
print 'Boolean array:', map(bool, l3)

l4 = [1, 2, 3]

print 'l3:', l3
print 'l4:', l4

for el3, el4 in zip(l3, l4): 
    print el3, el4

l5 = l3 + l4
print l5

for el3, el4, el5 in zip(l3, l4, l5): 
    print el3, el4, el5

def add(l, r): 
    try:
        return l * r
    
    except TypeError: 
        # Addition of `None` type is not defined
        return None 

def is_None(l, r): 
    return l is None or r is None

l5 = [5, 4, 3, 2, 1]

print map(add, l4, l5)
print map(is_None, l4, l5)

for index, value in enumerate(l1): 
    print index, value

for index, (el3, el4, el5) in enumerate(zip(l3, l4, l5)): 
    print index, (el3, el4, el5)

for index, (el3, el4, el5) in enumerate(zip(l3, l4, l5), start=100): 
    print index, (el3, el4, el5)





