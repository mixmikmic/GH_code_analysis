result,item = 0,0
while item < 4:
    item += 1
    result += item
    if result > 3: break 

print(result)

grades = [98,67,88,79,90]
for grade in grades:
    print('grade: ',grade)

# we can use 'enumerate' for access to both the integer index and value
names = ['Mal', 'Zoe', 'Wash', 'Inara', 'Jayne', 'Kaylee', 'Simon', 'River', 'Book']
for idx,name in enumerate(names):
    print('%d\t%s' % (idx,name))

n = ['   one   ', '   two   ', '   three   ']
stripped = [num.strip() for num in n]
print(stripped)

n = ['   one   ', '   two   ', '   three   ']
stripped = []
for cn in n:
    stripped.append(cn.strip())
print(stripped)

# we can even throw in control
odd = [i for i in range(10) if i % 2]
print(odd)

# they can be nested at any level
cnum = [complex(x,y) for x in range(5) for y in range(5) if abs(complex(x,y)) < 3]
print(cnum)

prime=[]
n=50
for i in range(1,n+1):
    y=0
    for x in range(2,i):
        if i%x==0:
            y=1
    if y==0:
        prime.append(i)
print(prime)

# %load data/primes.py
# generate all the non-primes (easy)
non_primes = [j for i in range (2,8) for j in range(i*2,50,i)]

# exclude anything but primes
primes = [x for x in range(2,50) if x not in non_primes]
print(primes)

# In a single line:
print([x for x in range(2,50) if not [t for t in range(2,x) if not x%t]])

from IPython.display import Image
Image(url='https://raw.githubusercontent.com/agdelma/IntroCompPhysics/master/Notebooks/data/function.png')

# let's define this function
def add(a,b):
    '''Add two numbers.'''
    c = a+b
    return c

def add_one_liner(a,b): return a+b

# The first comment line is used to describe the function when getting help.
help(add)

# let's use it!
add(10,11)

# floats
add(1.0,1.3)

# strings
add('red','sox')

# but we can't mix types
add('red',1)

# since everything is a first class object in python, 
# we can assign our function to a variable
func = add
func(10,10)

# functions with list comprehensions
def cube(x): return x**3
cubes = [cube(x) for x in range(1,10)]
print(cubes)

# functions as filters
def odd(x): return (x%2)
odds = [i for i in range(1,10) if odd(i)]
print(odds)

# using a function
def square(x): 
    return x**2
[square(x) for x in range(5)]

# using a lambda
sq = lambda x: x**2
[sq(x) for x in range(5)]

# they can be used anywhere that a function is required
student_tuples = [('john', 'A', 22), ('jane', 'B', 19),('dave', 'B', 23)]
sorted(student_tuples, key=lambda student: student[2])   # sort by age



# %load data/leapyear.py
def leap_year(year):
    '''Determine if the supplied year is a leap year.'''
    if not(year % 400):
        return True
    elif not(year % 100):
        return False
    elif not(year % 4):
        return True
    else:
        return False

lyears = [year for year in range(1979,2017) if leap_year(year)]
print(lyears)



