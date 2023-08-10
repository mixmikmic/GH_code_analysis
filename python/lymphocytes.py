class Cell(object):
    ''' The mother of all the cells'''
    def __init__(self):
        ''' The constructor'''
        self.type = 'generic'
        self.place = 'wherever'
        self.state = 'undefined'
    def __str__(self):
        ''' When you call using print'''
        return "Type: %s Place: %s State: %s"%(self.type,self.place,self.state)
    def death(self):
        ''' Kill a cell'''
        print "I'ts the end of the world as we know it!"
        self.state='dead'

c = Cell()
print c
print c.place # The dot allows us to access to the class attributes and methods
print c.state
c.death()
print c.state

class lymphoid(Cell):
    def __init__(self):
        self.type = 'lymphoid progenitor'
        self.place = 'bone marrow'
        self.state = 'immature'

l = lymphoid()
print l
print l.state
l.death() # lymphoid has inherited the method "death" from its ancestor
print l.state

lymphoid_list = [lymphoid() for i in range(10)]

from random import random,randint
print random()
print randint(1,8)
from math import log
log(1.)

class lymphoid(Cell):
    def __init__(self):
        self.type = 'lymphoid progenitor'
        self.wheretogo = 0.6 # Probability of staying at the bone marrow or go to the thymus
        self.place = 'bone marrow'
        self.state = 'immature'
    def move(self):
        if random()>=0.6:
            self.place = 'thymus'
            

class APC(Cell):
    ''' Antigen presenting cell'''
    def __init__(self,place='thymus'):
        self.type = 'Antigen presenting cell'
        self.place = place
        self.pMHC = random()
        self.HLA = randint(0,1)
        self.state = 'undefined'
    

l = lymphoid()
print l
l.move()
print l

class CellPopulation(object):
    def __init__(self):
        self.thelist = [] # To store the elements of the population
        self.len = 0 # The length 
    
    def add(self,newcell):
        self.thelist.append(newcell)
        self.len += 1 # we could do simply len(self.thelist), but that takes time
    
    def __str__(self):
        ''' Define the output when printing '''
        out = ""
        for i in range(self.len):
            out += self.thelist[i].__str__()+"\n" # Concatenate the outputs of every cell.
        return out
    
    def killcell(self,i):
        self.thelist[i].state ='dead'
        self.updatecell(i)
    
    def updatecell(self,i):
        if self.thelist[i].state=='dead':
            self.thelist[i]=self.thelist[-1] # A good-old trick, replace the current by the last...
            self.thelist.pop(-1) # ... and pop the last element (this is the fastest way to kill)
            self.len -= 1 # Update the count

cp = CellPopulation()
[cp.add(lymphoid()) for i in range(5)];

print cp
cp.killcell(4)
print cp

class tlymph(lymphoid):
    def __init__(self):
        self.type = 'T lymphocyte'
        self.place = 'thymus'
        self.state = 'beta_arrangement'
        self.TCR = random()
        
    def checkpoint(self,APC):
        #print 'pre-state: ',self.state
        if self.state =='beta_arrangement':
            self.state = 'alpha_arrangement'
            if APC.pMHC > self.TCR:
                self.state = 'dead'
                
        if self.state=='alpha_arrangement':
            if APC.HLA == 1:
                self.state = 'Killer T-cell'
            else:
                self.state = 'Helper T-cell'
        #print 'post-state: ',self.state

t = tlymph()
a = APC()
b = APC()
print t
print a
print b

print t
t.checkpoint(a)
print t
t.checkpoint(b)
print t

tpop = CellPopulation()
[tpop.add(tlymph()) for i in range(10)];
apop = CellPopulation()
[apop.add(APC()) for i in range(10)];

class Scheduler(object):
    def __init__(self):
        self.time = 0
        self.rate = 100 # total probability of encounter per day
    def update(self):
        ti = randint(0,tpop.len-1) # Pick one tlymph
        ai = randint(0,apop.len-1) # Pick one APC
        tpop.thelist[ti].checkpoint(apop.thelist[ai]) # Let the tlymph meet an APC
        tpop.updatecell(ti) # Update its state
        self.time += -log(random())/self.rate # A quick and dirty Gillespie algorithm

simulation = Scheduler()
for item in tpop.thelist: # Print the TCR "spectrum"
    print item.TCR
print "Simulation time=",simulation.time

for steps in range(100): # Run 100 times
    simulation.update()
print "Simulation time=",simulation.time

print tpop

for item in tpop.thelist:
    print item.TCR

