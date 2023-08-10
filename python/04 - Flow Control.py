x = 3 
y = 4
if x < y:
    print("y is greater")    # code block

a = 5 
b = 5
if a > b:
    print("a is greater")      # first code block
elif a == b:
    print("values are equal")  # alternative code block

v = 9
w = 10
if v > w:
    print("v is greater")      # first branch
elif v == w:
    print("both values are equal")  # alternative branch
else: 
    print("last resort, w must be greater")      # last resort

x = 5
while x < 10:                    # header line with condition
    print("x is currently", x)   # code block
    x = x + 1

y = 10
while y < 4:
    print(y)    # this code block will not be executed

z = 1
while z < 8:
    print("z is",z)
    if z%3 == 0:
        print("breaking out of loop when z is",z)
        break
    z += 1

vals = [2.5, 4.2, 3.1, 6.7, 8.9]
for v in vals:
    print(v)

mylist = ["UCD",2020,True,0.0123]
for x in mylist:
    print("Next item is",x)

countries = set(["Ireland","France","Germany","Italy"])
for country in countries:
    print(country)

capitals = { "Italy":"Rome", "Ireland":"Dublin", "Germany":"Berlin" }
for country in capitals:
    print(country, "=", capitals[country])

# Iterate over a sequence starting at 0 and ending before 4:
for i in range(4):
    print(i)

# Iterate over a sequence starting at 3 and ending before 8:
for i in range(3,8):
    print(i)

# Iterate over a sequence starting at 5, ending before 20, and incrementing by 4 at each step:
for i in range(5,20,4):
    print(i)

# Start at 40, ending before 20, and decrementing by 5 each time (i.e adding -5):
for i in range(40,20,-5):
    print(i)

list(range(5,20,4))

