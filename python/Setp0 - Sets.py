# Creating an empty set
languages = set()
print type(languages), languages

languages = {'Python', 'R', 'SAS', 'Julia'}
print type(languages), languages

# set of mixed datatypes
mixed_set = {"Python", (2.7, 3.4)}
print type(mixed_set), languages

print list(languages)[0]
print list(languages)[0:3]

# initialize a set
languages = {'Python', 'R'}
print(languages)

# add an element
languages.add('SAS')
print(languages)

# add multiple elements
languages.update(['Julia','SPSS'])
print(languages)

# add list and set
languages.update(['Java','C'], {'Machine Learning','Data Science','AI'})
print(languages)

# remove an element
languages.remove('AI')
print(languages)

# discard an element, although AI has already been removed discard will not throw an error
languages.discard('AI')
print(languages)

# Pop will remove a random item from set
print "Removed:", (languages.pop()), "from", languages

# initialize A and B
A = {1, 2, 3, 4, 5}
B = {4, 5, 6, 7, 8}

# use | operator
print "Union of A | B", A|B

# alternative we can use union()
A.union(B)

# use & operator
print "Intersection of A & B", A & B

# alternative we can use intersection()
print A.intersection(B)

# use - operator on A
print "Difference of A - B", A - B

# alternative we can use difference()
print A.difference(B)

# use ^ operator
print "Symmetric difference of A ^ B", A ^ B

# alternative we can use symmetric_difference()
A.symmetric_difference(B)

# Return a shallow copy of a set
lang = languages.copy()
print languages
print lang

# initialize A and B
A = {1, 2, 3, 4, 5}
B = {4, 5, 6, 7, 8}

print A.isdisjoint(B)   # True, when two sets have a null intersection
print A.issubset(B)     # True, when another set contains this set
print A.issuperset(B)   # True, when this set contains another set
sorted(B)               # Return a new sorted list
print sum(A)           # Retrun the sum of all items
print len(A)           # Return the length 
print min(A)           # Return the largest item
print max(A)           # Return the smallest item

