print 'Hello World!'

i = 4  #int

type(i)

f = 4.1  #float

type(f)

b = True  #boolean variable

s = "This is a string!"

print s

l = [3,1,2]  #list

print l

d = {'foo':1, 'bar':2.3, 's':'my first dictionary'}  #dictionary

print d

print d['foo']  #element of a dictionary

n = None  #Python's null type

type(n)

print "Our float value is %s. Our int value is %s. %s" % (f,i,6)  #Python is pretty good with strings

if i == 1 and f > 4:
    print "The value of i is 1 and f is greater than 4."
elif i > 4 or f > 4:
    print "i or f are both greater than 4."
else:
    print "both i and f are less than or equal to 4"

print l

for e in l:
    print e

counter = 6
while counter < 10:
    print counter
    counter += 1

def add2(x):
    y = x + 2
    return y

i = 5

add2(i)

square = lambda x: x*x

print square(5)

