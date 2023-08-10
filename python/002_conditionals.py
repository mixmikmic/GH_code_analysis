# if..else
v1 = 5
if v1 == 5:
    print (v1)
else:
    print ("v1 is not 5")

# if..elif..else
s1 = "Jennifer"
s2 = "loves"
s3 = "Python"
if s1 == "Python":
    print ("s1 is Python")
elif s2 == "Jennifer":
    print ("s2 is Jennifer")
elif s1 == "loves":
    print ("s1 is loves")
else:
    print ("Jennifer loves Python!")

# One liner
v1 = 5
x = 10 if v1 == 5 else 13
print (x)

# Let's see the conditionals available
v1 = "Jennifer"
v2 = "Python"
v3 = 45
v4 = 67
v5 = 45

# Test for equality
print (v1 == v2)

# Test for greater than and greater than equal
print (v4 > v3)
print (v5 >= v2)

# Test for lesser than and lesser than equal
print (v4 < v3)
print (v5 <= v2)

# Inequality
print (v1 != v2)

# Note:
v1 = 45
v2 = "45"
print (v1 == v2) # False
print (str(v1) == v2) # True

# Ignore case when comparing two strings
s1 = "Jennifer"
s2 = "jennifer"

print (s1 == s2) # False
print (s1.lower() == s2.lower()) # True
# OR
print (s1.upper() == s2.upper()) # True

# Checking multiple conditions 'and' and 'or'
v1 = "Jennifer"
v2 = "Python"

# 'and' -> evaluates true when both conditions are True
print (v1 == "Jennifer" and v2 == "Python")
# 'or' -> evaluates true when any one condition is True
print (v1 == "Python" or v2 == "Python")

s1 = "Jennifer"
s2 = "Python"

print (s1 > s2) # True -> since 'Jennifer' comes lexographically before 'Python'

# Check whether a value is in a list -> 'in'
l1 = [23, 45, 67, "Jennifer", "Python", 'A']

print (23 in l1)
print ('A' in l1)
print ("Python" in l1)
print (32 in l1)

# Putting it together
l1 = [23, 1, 'A', "Jennifer", 9.34]

# This is True, so the other statements are not checked
if 23 in l1 and 'B' not in l1: # Note: use of 'not'
    print ("1")
elif 23 >= l1[0]: # True
    print ("2")
elif 2.45 < l1[-1]: # True
    print ("3")

# Checking if list is empty
l1 = []
l2 = ["Jennifer"]

if l1:
    print (1)
elif l2:
    print (2)

