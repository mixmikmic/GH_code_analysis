from rmgpy.rmg.model import CoreEdgeReactionModel
from rmgpy.rmg.output import saveOutputHTML
from rmgpy.chemkin import loadChemkinFile
import os

path = os.path.abspath('data/pdd1010/')
chemkinPath= os.path.join(path, 'chem_annotated.inp')
dictionaryPath = os.path.join(path, 'species_dictionary.txt')
model = CoreEdgeReactionModel()
model.core.species, model.core.reactions = loadChemkinFile(chemkinPath,dictionaryPath, readComments = True)
outputPath = os.path.join(path, 'output.html')
speciesPath = os.path.join(path,'species')
if not os.path.isdir(speciesPath):
    os.makedirs(speciesPath)
saveOutputHTML(outputPath, model)



