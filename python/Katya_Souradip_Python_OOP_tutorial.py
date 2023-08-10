class Dog:
    def __init__(self, name):
        self.name = name
        self.legs = 4
        
    def bark(self):
        if self.legs is 4:
            print("Woof!")
        else:
            print("OWWWW")
            
rover = Dog("rover")
rover.bark()

def maim(self):
    self.legs = 0
Dog.maim = maim
rover.maim()
rover.bark()

def maim2(self):
    self.legs = 0
rover.maim2 = maim2
rover.maim2()
rover.bark()

rover.legs = 4
rover.bark()

class AlienPoodle(Dog):
    def __init__(self, name):
        Dog.__init__(self, name)
        self.legs = 8
        
fluffy = AlienPoodle("fluffy")
fluffy.bark()

class AlienPoodle(Dog):
    def __init__(self, name):
        Dog.__init__(self, name)
        
    def bark(self):
        print("Greetings earthlings *wags tail*")
        
fluffy = AlienPoodle("fluffy")
fluffy.bark()

fluffy.maim()
print(fluffy.legs)

class Chihuahua(Dog):
    def breed():
        print("Chiuahaha")
        
bailey = Chihuahua("bailey")
Chihuahua.breed()

bailey.breed()

Dog.bark(bailey)

class Cat():
    def __init__(self, name):
        self.tail = 1
        self.name = name
    
    nose = 1
    foods = ["tuna", "cream"]
    
bob = Cat("bob")
print(bob.nose)
print(bob.foods)

bob.nose = 0.5
bob.foods[0] = "salmon"
charlie = Cat("charlie")
print(charlie.nose)
print(charlie.foods)

