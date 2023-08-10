





class Vehicle():
    def __init__(self, name):
        self.name = name
        print("I am a {}.".format(self.name))
        
    def drive(self, mileage):
        print("I drove {} miles.".format(mileage))

class Car(Vehicle):
    def drive(self, mileage, passengers):
        print("I have {} passengers.".format(passengers))
        super(Car, self).drive(mileage)
    
class BMW(Car):
    def __init__(self, model, year):
        self.model = model
        self.year = year
        super(BMW, self).__init__(model)

bmw = BMW("Roadster", 2016)
bmw.drive(10, 3)









import numpy as np

list1 = np.random.random(10).tolist()
list2 = np.random.randint(100, 10).tolist()

data_structure = enumerate(zip(list1, list2))  # What is this?



x = 2

def f(x):
    y = x ** 2
    x = x ** 2
    return y

y = f(x)
x = f(y)

# What is x?
# What is y?

def g(x, y):
    x[0] *= x[1]
    z = y.copy()
    z[0] *= x[1]
    return z

x = [1, 2]
y = [3, 4]
x = g(y, x)

# What are x and y?

