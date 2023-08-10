get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import pickle

from astropy.table import Table

def makeTable(effDict, effTable, effKeys):
    for filename in effDict:
        row = np.zeros(len(effKeys))
        for i,key in enumerate(effKeys):
            try:
                row[i] = effDict[filename][key] 
            except KeyError:
                row[i] = 0.
        effTable.add_row(row)
    return effTable

def plotStuff(OldTable, NewTable, key, title,scale=100000.):
    for angle in set(OldTable['Cos']):
        maskOld = OldTable['Cos'] == angle
        maskNew = NewTable['Cos'] == angle
        plt.plot(NewTable[maskNew]['MeV'],
                 NewTable[maskNew][key]/scale,
                's', label = 'new', alpha=0.5)
        plt.plot(OldTable[maskOld]['MeV'],
                 OldTable[maskOld][key]/scale,
                 '.', label = 'old')
        plt.title('Cos{:0.1f}'.format(angle))
        plt.xscale('log')
        plt.xlabel('Energy [MeV]')
        plt.ylabel(title)
        plt.legend()
        plt.show()

cosimaKeys = ('Cos','MeV','numberOfSimulatedEvents', 'numberOfTriggers')
cosimaTypes = ('f4','f4','i4','i4')

revanKeys = ('Cos',
             'MeV',
             'MainTrigger',
             'MainVetoSide',
             'Not triggered events',
             'Number of triggered events',
             'Number of vetoed events',
             'MainVetoTop')
revanTypes = ('f4', "f4",'i4', 'i4', 'i4', 'i4', 'i4', 'i4')

SimPath = "/data/slag2/ComPair/Simulations/"
OldPath = SimPath + "AMEGO_20161031/"
NewPath = SimPath + "AMEGO_20170524/"

revanEffNew = pickle.load(open( NewPath + "revanEff.p", "rb" ))
cosimaEffNew = pickle.load(open( NewPath + "cosimaEff.p", "rb"))
revanEffOld = pickle.load(open( OldPath + "revanEff.p", "rb" ))
cosimaEffOld = pickle.load(open( OldPath + "cosimaEff.p", "rb"))

cosimaTableNew = Table(names=cosimaKeys, dtype=cosimaTypes)
cosimaTableOld = Table(names=cosimaKeys, dtype=cosimaTypes)

revanTableNew = Table(names=revanKeys, dtype=revanTypes)
revanTableOld = Table(names=revanKeys, dtype=revanTypes)

revanTableNew = makeTable(revanEffNew,revanTableNew,revanKeys)

revanTableOld = makeTable(revanEffOld,revanTableOld,revanKeys)

cosimaTableNew = makeTable(cosimaEffNew,cosimaTableNew,cosimaKeys)
cosimaTableOld = makeTable(cosimaEffOld,cosimaTableOld,cosimaKeys)

plotStuff(revanTableOld,revanTableNew,'Number of triggered events','Revan Triggered Events [%]')

plotStuff(revanTableOld,revanTableNew,'Number of vetoed events','Number of vetoed events [%]')

plotStuff(cosimaTableOld,cosimaTableNew,'numberOfSimulatedEvents','Cosima Simulated Events', scale=1.)



