mylist = [1, 2, 3]
for i in mylist:
    print (i)

mylist = [x*x for x in range(3)]
for i in mylist:
    print (i)

# Build and return a list
def firstn(n):
    num, nums = 0, []
    while num < n:
        nums.append(num)
        num += 1
    return nums

sum_of_first_n = sum(firstn(1000000))
print sum_of_first_n

mygenerator = (x*x for x in range(3))
for i in mygenerator:
    print (i)
    

# list comprehension
doubles = [2 * n for n in range(50)]

# same as the list comprehension above
doubles = list(2 * n for n in range(50))

# Using the generator pattern (an iterable)
class firstn(object):
    def __init__(self, n):
        self.n = n
        self.num, self.nums = 0, []

    def __iter__(self):
        return self

    def next(self):
        if self.num < self.n:
            cur, self.num = self.num, self.num + 1
            return cur
        else:
            raise StopIteration()

            
sum_of_first_n = sum(firstn(1000000))
print sum_of_first_n

def gensquares(N):
    for i in range(N):
        yield i ** 2 # Resume here later

for i in gensquares(5): # Resume the function 
    print(i) # Print last yielded value

# a generator that yields items instead of returning a list
def firstn(n):
    num = 0
    while num < n:
        yield num
        num += 1

sum_of_first_n = sum(firstn(1000000))

def createGenerator():
    mylist = range(3)
    for i in mylist:
        yield i*i

mygenerator = createGenerator() # create a generator
print('Object type:', mygenerator) # mygenerator is an object!


for i in mygenerator:
    print (i)



