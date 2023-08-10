my_str = "Hello World!"
print(dir(my_str))

# l = ["hello", "hola", "world"]
# for each in l:
#     if each.startswith("h"):
#         print(each)

x = "Hello World"
print(x.split(" "))

#!/usr/bin/python3
class MyFirstClass:
    """This is a doc string abc"""
    pass

# Insatances / objects
a = MyFirstClass()
b = MyFirstClass()
print(a.__doc__)

print(dir(object))

print(dir(a))
# print(a.__dict__)
# print(a.__doc__)
#print(a.__hash__())
#print(a.__sizeof__())
#print(a.__str__())

class Point:
    pass

p1 = Point()
p2 = Point()

p1.x = 5
p1.y = 4

p2.x = 3
p2.y = 6

print(p1.x, p1.y)
print(p2.x, p2.y)

class Point:
    def reset(self):
        self.x = 0
        self.y = 0

p = Point()
p2 = Point()
p.x = 5
p.y = 4
print(p.x, p.y)
p.reset()
p2.reset()
print(p.x, p.y)

# Invoking method statically
class Point:
    def reset(self):
        self.x = 0
        self.y = 0

p = Point()
Point.reset(p)
print(p.x, p.y)

class Employee:
    """ This is a doc string for Employee class"""
    def __init__(self, first, last, pay):
        # here self is the automatically invoked instance to the method
        self.first = first
        self.last = last
        self.pay = pay
        # email can be derived from the "first" and "last" name
        self.email = self.first+"."+self.last+"company.com"

    # a method that returns fullname. self is a must for any method
    def fullname(self):
        return "{0} {1}".format(self.first, self.last)

# instances with data
emp_1 = Employee("Sagar", "Giri", 50000)
emp_2 = Employee("Test", "User", 60000)

# print(emp_1.__dict__)
# print(dir(emp_1))
# print(emp_2.email)

# While calling a methods, don't forget the parenthesis
print(emp_1.fullname())
print(emp_2.fullname())

# calling method directly from class.
# To do so, we need to pass the instance we created.
# Remember the self we added in the method :)
print("Directly from class---> ",Employee.fullname(emp_1))
print("Directly from class---> ",Employee.fullname(emp_2))

