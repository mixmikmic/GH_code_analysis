from DataModelDict import DataModelDict as dmd

schema = dmd(open('kolskybar.xml','r'))
table = schema.find('stressStrain')

distable = []
for row in table['rows'].iterlist('row'):
        disrow = []
        for column in row.iterlist('column'):
                disrow.append(column['#text'])
        distable.append(disrow)

del distable[0]
distable = np.array(distable)

print distable

from DataModelDict import DataModelDict as dmd

schema = dmd(open('kolskybar.xml','r'))
table = schema.find('stressStrain')

#print table.keys()
rows= table['rows']
for row in rows.iteraslist('row'):
    print row
    break

import numpy as np
import matplotlib.pyplot as plt

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

p_alpha = 1
volume=50
fig, ax = plt.subplots(figsize=(12, 9))

# plots first col vs second because stress vs strain
ax.plot(distable[:,0],distable[:,1],lw=3) 
ax.tick_params(axis='x', labelsize=25, pad = 10)
ax.tick_params(axis='y', labelsize=25, pad = 10)


ax.grid(True)
fig.tight_layout()


plt.show()

from irreversible_stressstrain import StressStrain
import matplotlib.pyplot as plt

model = StressStrain('kolskybar.xml','xml')
fig, ax = plt.subplots(figsize=(12, 9))
distable = model.get_experimental_data()

# plots first col vs second because stress vs strain
ax.plot(distable[:,0],distable[:,1],lw=3) 
ax.tick_params(axis='x', labelsize=25, pad = 10)
ax.tick_params(axis='y', labelsize=25, pad = 10)


ax.grid(True)
fig.tight_layout()


plt.show()

from irreversible_stressstrain import StressStrain

model2 = StressStrain('ref/HSRS/326')
print model2.mcfunc((-341.213647127, 6.68840882182))

model3 = StressStrain('ref/HSRS/222')
print model3.mcfunc((-160.27073606,  138.1929225))

