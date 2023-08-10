from rmgpy.rmg.main import RMG, CoreEdgeReactionModel
from rmgpy.data.rmg import RMGDatabase, database
from rmgpy.rmg.model import Species
from rmgpy.molecule import Molecule
from rmgpy import settings
import os

# set-up RMG object
rmg = RMG()
rmg.reactionModel = CoreEdgeReactionModel()

# load kinetic database and forbidden structures
rmg.database = RMGDatabase()
path = os.path.join(settings['database.directory'])

# forbidden structure loading
database.loadForbiddenStructures(os.path.join(path, 'forbiddenStructures.py'))
# kinetics family Disproportionation loading
database.loadKinetics(os.path.join(path, 'kinetics'),                           kineticsFamilies=['R_Addition_MultipleBond'])

spc = Species().fromSMILES("O=C[C]=C")
print spc.molecule[0].toSMILES()
print spc.molecule[0].toAdjacencyList()

newReactions = []
spc.generateResonanceIsomers()
newReactions.extend(rmg.reactionModel.react(database, spc))

# try to pick out the target reaction I want to show
mol_H = Molecule().fromSMILES("[H]")
mol_C3H2O = Molecule().fromSMILES("C=C=C=O")
for rxn in newReactions:
    reactants = rxn.reactants
    products = rxn.products
    rxn_specs = reactants + products
    for rxn_spec in rxn_specs:
        if rxn_spec.isIsomorphic(mol_H):
            for rxn_spec1 in rxn_specs:
                if rxn_spec1.isIsomorphic(mol_C3H2O):
                    for rxn_spec in rxn_specs:
                        rxn_spec.label = rxn_spec.molecule[0].toSMILES()
                    print rxn
                    print rxn.template

rmg.reactionModel.processNewReactions(newReactions, spc, None)
for rxn in rmg.reactionModel.edge.reactions:
    # try to pick out the target reaction I want to show
    reactants = rxn.reactants
    products = rxn.products
    rxn_specs = reactants + products
    for rxn_spec in rxn_specs:
        if rxn_spec.isIsomorphic(mol_H):
            for rxn_spec1 in rxn_specs:
                if rxn_spec1.isIsomorphic(mol_C3H2O):
                    print rxn
                    print rxn.template

spc.molecule = list(reversed(spc.molecule))

newReactions = []
newReactions.extend(rmg.reactionModel.react(database, spc))

mol_H = Molecule().fromSMILES("[H]")
mol_C3H2O = Molecule().fromSMILES("C=C=C=O")
for rxn in newReactions:
    reactants = rxn.reactants
    products = rxn.products
    rxn_specs = reactants + products
    for rxn_spec in rxn_specs:
        if rxn_spec.isIsomorphic(mol_H):
            for rxn_spec1 in rxn_specs:
                if rxn_spec1.isIsomorphic(mol_C3H2O):
                    for rxn_spec in rxn_specs:
                        rxn_spec.label = rxn_spec.molecule[0].toSMILES()
                    print rxn
                    print rxn.template

# set-up RMG object
rmg_new = RMG()
rmg_new.reactionModel = CoreEdgeReactionModel()

rmg_new.reactionModel.processNewReactions(newReactions, spc, None)

for rxn in rmg_new.reactionModel.edge.reactions:
    reactants = rxn.reactants
    products = rxn.products
    rxn_specs = reactants + products
    for rxn_spec in rxn_specs:
        if rxn_spec.isIsomorphic(mol_H):
            for rxn_spec1 in rxn_specs:
                if rxn_spec1.isIsomorphic(mol_C3H2O):
                    print rxn
                    print rxn.template

spc = Species().fromSMILES("O=C[C]=C")
print spc.molecule[0].toSMILES()
print spc.molecule[0].toAdjacencyList()

# to run the code below you should checkout the `edge_inchi_rxn` branch
from rmgpy.rmg.model import InChISpecies
ispc = InChISpecies(spc)

spc_new = Species(molecule=[Molecule().fromAugmentedInChI(ispc.getAugmentedInChI())])
print spc_709_new.molecule[0].toSMILES()
print spc_709_new.molecule[0].toAdjacencyList()

