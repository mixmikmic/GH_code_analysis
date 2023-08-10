import os
import numpy as np
import scipy.integrate as integrate
import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')
import distributions2 as d2

mainDir = "/home/gf/src/Python/Python-in-the-lab/Bk"
dcoll = d2.DistCollector(mainDir)

dcoll.plot("S") # Isn't it beatifull???

dcoll.plot("T") # Isn't it beatifull???

dcoll.plot("t") # Ahhhhhhhhhhhhhhhhhhhhhhh

import distributions3 as d3

dcoll = d3.DistCollector(mainDir)

dcoll.plot('E')

dcoll.plot('t')

names = {'S': 'size', 'T': 'duration', 'E': 'energy', 'v': 'velocity'}
class Labels:
    def __init__(self, dis_type):
        try:
            lb = names[dis_type]
        except KeyError:
            print("Distribution type %s not valid" % dis_type)
            sys.exit()
        name = names[dis_type]
        self.xlabel, self.ylabel, self.title = "{0} {1}".format(name, dis_type), "P({})".format(dis_type), "{} distribution".format(name)

lbS = Labels('S')

lbS.title, lbS.xlabel, lbS.ylabel

# As a better alternative...
class Labels:
    def __init__(self, dis_type):
        try:
            lb = names[dis_type]
        except KeyError:
            print("Distribution type %s not valid" % dis_type)
            sys.exit()
        self.name = names[dis_type]
        self.dis_type = dis_type
        
    @property
    def xlabel(self):
        return "{0} {1}".format(self.name, self.dis_type)
    
    @property
    def ylabel(self):
        return "P({})".format(self.dis_type)
    
    @property
    def title(self):
        return "{} distribution".format(self.name)

lb = Labels('T')
lb.title, lb.xlabel, lb.ylabel

# You can instert this in the distributions3.py for instance



