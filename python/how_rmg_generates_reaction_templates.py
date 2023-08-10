get_ipython().magic('matplotlib inline')
from rmgpy.rmg.main import RMG, CoreEdgeReactionModel
from rmgpy.data.rmg import RMGDatabase, database
from rmgpy.data.base import ForbiddenStructureException
from rmgpy.molecule import Molecule
from rmgpy.species import Species
from IPython.display import display
import os

# set-up RMG object
rmg = RMG()
rmg.reactionModel = CoreEdgeReactionModel()

# load kinetic database and forbidden structures
rmg.database = RMGDatabase()
home_path = os.getenv('HOME')
path = os.path.join(home_path, 'Code', 'rmgmaster', 'RMG-database', 'input')

# forbidden structure loading
rmg.database.loadForbiddenStructures(os.path.join(path, 'forbiddenStructures.py'))
# kinetics family Disproportionation loading
rmg.database.loadKinetics(os.path.join(path, 'kinetics'),                           kineticsFamilies=['Disproportionation'])

# create reactants (molecules): C=CC=C and C=CCC
molA = Molecule().fromSMILES("C=CC=C")
molB = Molecule().fromSMILES("C=CCC")
reactants = [molA, molB]
print "molA:"
display(molA)
print "molB:"
display(molB)

# pick out the reacting family
families = rmg.database.kinetics.families
disprop_family = families['Disproportionation']

# map reactants to the reacting family's template reactants
family_template = disprop_family.reverseTemplate
mappingsA = disprop_family._KineticsFamily__matchReactantToTemplate(molA, family_template.reactants[1])
mappingsB = disprop_family._KineticsFamily__matchReactantToTemplate(molB, family_template.reactants[0])

# assign the labels to atoms based on first mapping
mapA = mappingsA[0]
molA.clearLabeledAtoms()
for atom, templateAtom in mapA.iteritems():
    atom.label = templateAtom.label

print "molA with first mapping: \n" + str(molA.toAdjacencyList())

# assign the labels to atoms based on third mapping
mapA = mappingsA[2]
molA.clearLabeledAtoms()
for atom, templateAtom in mapA.iteritems():
    atom.label = templateAtom.label

print "molA with third mapping: \n" + str(molA.toAdjacencyList())

# assign the labels to atoms based on first mapping
mapB = mappingsB[0]
molB.clearLabeledAtoms()
for atom, templateAtom in mapB.iteritems():
    atom.label = templateAtom.label
print "molB with first mapping: \n" + str(molB.toAdjacencyList())
# assign the labels to atoms based on third mapping
mapB = mappingsB[2]
molB.clearLabeledAtoms()
for atom, templateAtom in mapB.iteritems():
    atom.label = templateAtom.label
print "molB with third mapping: \n" + str(molB.toAdjacencyList())

# create product structures by applying some mapping combination `(mapA, mapB)`
print "mapping combination: 1st mapA and 1st mapB"
for mapA in mappingsA[:1]:
    for mapB in mappingsB[:1]:
        reactantStructures = [molA, molB]
        try:
            productStructures =             disprop_family._KineticsFamily__generateProductStructures(reactantStructures,                                                                       [mapA, mapB],                                                                       False)
        except ForbiddenStructureException:
            pass
        else:
            if productStructures is not None:
                rxn1 = disprop_family._KineticsFamily__createReaction(reactantStructures,                                                                      productStructures,                                                                      False)
                if rxn1: print rxn1

# create product structures by applying another mapping combination `(mapA, mapB)`
print "mapping combination: 3rd mapA and 3rd mapB"
for mapA in mappingsA[2:3]:
    for mapB in mappingsB[2:3]:
        reactantStructures = [molA, molB]
        try:
            productStructures =             disprop_family.            _KineticsFamily__generateProductStructures(reactantStructures,                                                        [mapA, mapB],                                                        False)
        except ForbiddenStructureException:
            pass
        else:
            if productStructures is not None:
                rxn3 = disprop_family.                _KineticsFamily__createReaction(reactantStructures,                                                 productStructures, False)
                if rxn3: print rxn3

print disprop_family.getReactionPairs(rxn1)
print disprop_family.getReactionTemplate(rxn1)

print disprop_family.getReactionPairs(rxn3)
print disprop_family.getReactionTemplate(rxn3)

