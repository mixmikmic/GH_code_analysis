class Pet(object):
    
    def __init__(self, name, species):
        self.name = name
        self.species = species

    def getName(self):
        return self.name

    def getSpecies(self):
        return self.species

    def __str__(self):
        return "%s is a %s" % (self.name, self.species)

mycat = Pet("Odafin", "Cat")

mycat.getName()

Pet.getName(mycat)

mycat.getSpecies()

Pet.getSpecies(mycat)

print mycat

class Cat(Pet):
    def __init__(self, name, fluffy):
        Pet.__init__(self, name, "Cat")
        self.fluffy = fluffy

    def isFluffy(self):
        return self.fluffy

pet1 = Pet("Max", "Cat")

cat1 = Cat("Max", True)

isinstance(pet1, Pet)

isinstance(pet1, Cat)

isinstance(cat1, Pet)

isinstance(cat1, Cat)

pet1.getName()

cat1.getName()

pet1.isFluffy()

cat1.isFluffy()

print pet1

print cat1

dir(Cat)

dir(cat1)

def props(cls):
  return [i for i in cls.__dict__.keys() if i[:1] != '_']

props(Pet)

props(Cat)

vars(pet1)

vars(cat1)

