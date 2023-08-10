1 != 1

if False:
    print 'no print'
elif 1 == 1:
    print 'not gonna happen'
else:
    print 'else print'

result = 1
print 'result == 1 value:', result == 1
if True:
    print("Best Match")
elif result <= 1:
    print("Close Enough")
else:
    print("This is Blasphemy!")

for i in [0,1,2]:
    print 'a'

range(10)

for i in range(10):
    print i

i = 2
while i >= -5:
    print("{}".format(i))
    i -= 1

xrange(5)

type(range(5))

# With start and stop
list(range(2, 20))

# With start, stop and step
list(range(2, 20, 2))

a = [1,2,3]
b = [1,2,3]
c = [a,b]

a == b

for i in c:
    print c
    print i 
    for j in i:
        print j

# This is not the best way.. but for the sake of completion of
# topic, this example is included.
arr = [range(3), range(3, 6)]
for lists in arr:
    for elem in lists:
        print(elem)

for i in range(1, 10):
    if i == 5:
        print('Condition satisfied')
        break
    print(i)  # What would happen if this is placed before if condition?

for i in range(1, 10):
    if i == 5:
        print('Condition satisfied')
        continue
        print("whatever.. I won't get printed anyways.")
    print(i)

for i in range(1, 10):
    if i == 5:
        print('Condition satisfied')
        pass
    print(i)

best = 11
for i in range(10):
    if i >= best:
        print("Excellent")
        break
    else:
        continue
else:
    print("Couldn't find the best match")

best = 9
for i in range(10):
    if i >= best:
        print("Excellent")
        break
    else:
        continue
else:
    print("Couldn't find the best match")

