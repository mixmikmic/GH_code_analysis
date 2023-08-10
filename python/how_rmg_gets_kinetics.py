get_ipython().magic('matplotlib inline')
from rmgpy.rmg.main import RMG, CoreEdgeReactionModel
from rmgpy.data.rmg import RMGDatabase, database
from rmgpy.data.base import ForbiddenStructureException
from rmgpy.molecule import Molecule
from rmgpy.species import Species
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
database.loadKinetics(os.path.join(path, 'kinetics'),                           kineticsFamilies=['Disproportionation'])

spcA = Species().fromSMILES("C=CC=C")
spcB = Species().fromSMILES("C=CCC")

newReactions = []
newReactions.extend(rmg.reactionModel.react(database, spcA, spcB))

rmg.reactionModel.kineticsEstimator = 'rate rules'

reaction0 = newReactions[0]
template = reaction0.template
degeneracy = reaction0.degeneracy
estimator = rmg.reactionModel.kineticsEstimator
kinetics, entry = reaction0.family.getKineticsForTemplate(template, degeneracy, method=estimator)

kinetics

reaction0 = newReactions[2]
template = reaction0.template
degeneracy = reaction0.degeneracy
estimator = rmg.reactionModel.kineticsEstimator
kinetics, entry = reaction0.family.getKineticsForTemplate(template, degeneracy, method=estimator)

kinetics

reaction0 = newReactions[6]
template = reaction0.template
degeneracy = reaction0.degeneracy
estimator = rmg.reactionModel.kineticsEstimator
kinetics, entry = reaction0.family.getKineticsForTemplate(template, degeneracy, method=estimator)

kinetics

rules = reaction0.family.rules

entry = rules.getRule(template)

entry.data.comment

