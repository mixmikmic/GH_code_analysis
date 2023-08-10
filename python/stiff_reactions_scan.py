from rmgpy.rmg.model import CoreEdgeReactionModel
from rmgpy.chemkin import loadChemkinFile, saveChemkinFile
import os
from IPython.display import display

# chemkin model name
mech = 'surm009'

path = os.path.abspath('../')
mechPath = path + '/data/' + mech
chemkinPath= mechPath + '/chem.inp'
dictionaryPath = mechPath + '/species_dictionary.txt'

model = CoreEdgeReactionModel()
model.core.species, model.core.reactions = loadChemkinFile(chemkinPath,dictionaryPath)

len(model.core.reactions)

T = 6.731500e+02 # K

P = 4e5 # Pa
R = 8.314
bimolecularThreshold = 5e12 # 1/M*sec
unimolecularThreshold = bimolecularThreshold * (P*R/T)/1000 # 1/sec
unimolecularThreshold/1e10

threshold1 = unimolecularThreshold/1e18
threshold2 = bimolecularThreshold
rxnList = []
for rxn in model.core.reactions:
    rm = False
    if len(rxn.products) == 1:
        containS = False
        for atm in rxn.products[0].molecule[0].atoms:
            if atm.isSulfur():
                containS = True
                break
        if not containS: 
            rxnList.append(rxn)
            continue
        reverseRate = rxn.generateReverseRateCoefficient()
        if reverseRate.getRateCoefficient(T)  > threshold1:
#             print "##########rxn: {}##############".format(rxn)
#             print "##########reverse rate: {}##############".format(reverseRate.getRateCoefficient(T))
            rm = True
# #             display(rxn)
    
    elif len(rxn.reactants) == 1:
        containS = False
        for atm in rxn.reactants[0].molecule[0].atoms:
            if atm.isSulfur():
                containS = True
                break
        if not containS: 
            rxnList.append(rxn)
            continue
        forwardRate = rxn.kinetics
        if forwardRate.getRateCoefficient(T)  > threshold1:
#             print "##########rxn: {}##############".format(rxn)
#             print "##########reverse rate: {}##############".format(forwardRate.getRateCoefficient(T))
            rm = True
#             display(rxn)
    else:
        containS = False
        atomList = []
        for spe in rxn.reactants + rxn.products:
            atomList = atomList + spe.molecule[0].atoms
        
        for atm in atomList:
            if atm.isSulfur():
                containS = True
                break
        if not containS: 
            rxnList.append(rxn)
            continue
        
        forwardRate = rxn.kinetics
        reverseRate = rxn.generateReverseRateCoefficient()
        if (forwardRate.getRateCoefficient(T)  > threshold2) or (reverseRate.getRateCoefficient(T) > threshold2):
            rm = True
        
    
    if not rm:
        rxnList.append(rxn)
print len(rxnList)

mech_rm = os.path.join(path, 'data', mech+'_rm')
if not os.path.exists(mech_rm):
    os.mkdir(mech_rm)

saveChemkinFile(os.path.join(mech_rm, 'chem.inp'), model.core.species, rxnList, verbose = True, checkForDuplicates=False)

