from rmgpy.rmg.main import RMG
from rmgpy.rmg.model import CoreEdgeReactionModel
from rmgpy import settings
from IPython.display import display

# Create input file
# Here you can change the thermo and reaction libraries, or restrict families.  
input_header = """
database(
    thermoLibraries = ['KlippensteinH2O2','primaryThermoLibrary','DFT_QCI_thermo','CBS_QB3_1dHR'],
    reactionLibraries = [],  
    seedMechanisms = [],
    kineticsDepositories = 'default', 
    #this section lists possible reaction families to find reactioons with
    kineticsFamilies = ['R_Recombination'],
    kineticsEstimator = 'rate rules',
)
"""

speciesList = """
# List all species you want reactions between
species(
    label='ethane',
    reactive=True,
    structure=SMILES("CC"),
)

species(
    label='H',
    reactive=True,
    structure=SMILES("[H]"),
)

species(
    label='butane',
    reactive=True,
    structure=SMILES("CCCC"),
)
"""

# Write input file to disk
inputFile = open('scratch/input.py','w')
inputFile.write(input_header)
inputFile.write(speciesList)
inputFile.close()

# Execute generate reactions
from rmgpy.tools.generate_reactions import *
rmg = RMG()
rmg = execute(rmg,'scratch/input.py','scratch')

from rmgpy.cantherm.output import prettify
for rxn in rmg.reactionModel.outputReactionList:
    display(rxn)
    print prettify(repr(rxn.kinetics))

