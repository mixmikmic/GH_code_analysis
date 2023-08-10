# -Example
import numpy as np

import matplotlib.pyplot as plt

#By Lukas
from DataModelDict import DataModelDict as dmd

text_file = open('kolskybar.xml', "r")
qdata = dmd(text_file)
print qdata.xml(indent=2)

table = qdata.find('stressStrain')
distable = []
for row in table['rows'].iterlist('row'):
        disrow = []
        for column in row.iterlist('column'):
                disrow.append(column['#text'])
        distable.append(disrow)

del distable[0]
distable = np.array(distable)

print distable


p_alpha = 1
volume=50
fig, ax = plt.subplots(figsize=(12, 9))

ax.plot(distable[:,0],distable[:,1],lw=3) # plots first col vs second because stress vs strain
ax.tick_params(axis='x', labelsize=25, pad = 10)
ax.tick_params(axis='y', labelsize=25, pad = 10)


ax.grid(True)
fig.tight_layout()


plt.show()


get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import graph_suite as plot
import test_suite
reload(test_suite)

from DataModelDict import DataModelDict as dmd

table = dmd(open('kolskybar.xml', "r")).find('stressStrain')

distable = []

for row in table['rows'].iteraslist('row'):
    
        disrow = []
        
        for column in row.iteraslist('column'):
                disrow.append(column['#text'])
        
        distable.append(disrow)

del distable[0] # gets rid of header
distable = np.array(distable)

plot.plot2D(distable,'strain','stress','Stress vs. Strain for Carbon Model')

data = np.loadtxt('ref/HSRS/22')

test_suite.plot2D(data,'stress','strain','HSRS - 22')

data = np.loadtxt('ref/HSRS/222')

test_suite.plot2D(data,'stress','strain','HSRS - 222')

data = np.loadtxt('ref/HSRS/326')

test_suite.plot2D(data,'stress','strain','HSRS - 326')

import test_suite as suite
import irreversible_stressstrain as model
reload(test_suite)
reload(model)

model.mcfunc((-100.,1.),'ref/HSRS/22')

from irreversible_stressstrain import mcfunc
import test_suite as suite
suite.minimize_suite(mcfunc, ['L-BFGS-B'], [-100.,1] )

import test_suite
from irreversible_stressstrain import StressStrain
reload(test_suite)

model = StressStrain('ref/HSRS/22')
test_suite.minimize_suite(model.mcfunc,['L-BFGS-B'], [-100.,1])

get_ipython().magic('matplotlib inline')
import numpy as np
import graph_suite as plot
import test_suite

plot.plot2D(test_suite.xml_parse('kolskybar.xml'))

