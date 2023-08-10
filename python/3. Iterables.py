for i in 'abc':
    print(i)

for each in range(3):
    print(each)

'a'*(10**100)

range(10**100)

for each in filter(lambda x: x % 2 == 0, range(5)):
    print(each)

for each in map(lambda x: x**2, range(5)):
    print(each)

for i in map(lambda x: x**2, filter(lambda x: x % 2 == 0, range(5))):
    print(i)

for i in filter(lambda x: x > 5, map(lambda x: x**2, range(5))):
    print(i)

