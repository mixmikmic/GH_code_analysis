#boolean data type
a = True
b = False

print(type(a))

#Relational Operators

x = 10
y = 20

print(x>y)
print(x<y)
print(x>=y)
print(x<=y)
print(x==y)
print(x!=y)

#Logical Operators

x = 10
y = 20
z = 30

print(x>y and x>z)
print(x<y or x>z)

#if else

if x>y and x>z:
    print(x)
elif y>x and y>z :
    print(y)
else :
    print(z)

if True or True:
    if False and True or False:
        print('A')
    elif False and False or True and True:
        print('B')
    else:
        print('C')
else:
     print('D')

#While Loop

n = int(input())
i = 1
while i<=n :
    print(i, end = " ")
    i = i+1
print()
print("Done")

#For loop and range

n = int(input())

for i in range(1,n+1):
    print(i)

n = int(input())

for i in range(n+1):
    print(i)

n = int(input())

for i in range(0, n+1, 2):
    print(i)

#IsPrime

n = int(input())
isPrime = True

for i in range(2,n):
    if(n % i == 0):
        isPrime = False
        break

if isPrime:
    print("Prime")
else:
    print("Not a prime Number")

#Fast Iterations in strings and tuples

s = "abcde"
for c in s:
    print(c)
    
    
t = (1,2,3,4,5)
for b in t:
    print(b)

x = 'abcd'
for i in range(len(x)):
   print(x)
   x = 'a'

n = int(input())
i=1
while (i<n+1):
    j = 1
    while (j<i+1):
        print(j, end = "")
        j +=1
        
    j = i+1
    while (j < 2*n-i+1):
        print(" ", end = "")
        j +=1

    j = i
    while (j>0):
        print(j, end = "")
        j -=1
    
    print()
    i +=1
    
    

#Reverse each word individually 

s = input()

si = 0
i = 0
for i in range(0, len(s)) :
    if s[i] == " ":
        ei = i-1
        while j>=si:
            print(s[j])
            j -=1
        print()    
        si = i+1    
        

