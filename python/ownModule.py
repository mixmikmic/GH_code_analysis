import ownModuleA

import ownModuleB

print(ownModuleB.variable)
ownModuleB.help()

get_ipython().magic('pinfo ownModuleB')

import ownModuleC

print(ownModuleC.random100())
print(ownModuleC.random100())
print(ownModuleC.random100())
print(ownModuleC.random100())

ownModuleC.nessie()

ownModuleC.messy.irish_nessie()

import monsters

monsters.loch

monsters.loch.nessie

monsters.loch.nessie.nessie

get_ipython().magic('pinfo monsters.loch.nessie')

monsters.loch.__Nessie__

nessieRed = monsters.loch.nessie('Andy','red')

nessieRed

print(nessieRed)
print(nessieRed.name)
print(nessieRed.color)
print(nessieRed.length)
nessieRed.say_hi()

nessieGold = monsters.loch.nessie('Matthias','yellow', length=3)
nessieGold.say_hi()

nessieGold.color='blue'
nessieGold.say_hi()

names = ['Jodoc', 'Gunhild', 'Basajaun', 'Marius', 'Aldebrand', 'Oghenekevwe', 'Mi-Gyeong', 'Edorta']
colors = ['grey', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']
creatures = []

for n,c in zip(names, colors):
    ness = monsters.loch.nessie(n,c)
    ness.say_hi()
    creatures.append(ness)

# get out by using 'raise SystemExit'
ness.debug()

for c in creatures: print(c.name)



