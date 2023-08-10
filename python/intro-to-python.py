import h2o
import os
import tabulate
import operator 
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

h2o.init()

airlines_hex = h2o.import_file(path = os.path.realpath("../data/allyears2k.csv"),
                               destination_frame = "airlines.hex")
airlines_hex.describe()

# Set predictor and response variables
myY = "IsDepDelayed"
myX = ["Dest", "Origin", "DayofMonth", "Year", "UniqueCarrier", "DayOfWeek", "Month", "Distance"]

# GLM - Predict Delays
glm_model = H2OGeneralizedLinearEstimator(
    family = "binomial",standardize = True, solver = "IRLSM",
    link = "logit", alpha = 0.5, model_id = "glm_model_from_python" )
glm_model.train(x               = myX,
               y               = myY,
               training_frame  = airlines_hex)

print "AUC of the training set : " + str(glm_model.auc())
# Variable importances from each algorithm
# Calculate magnitude of normalized GLM coefficients
glm_varimp = glm_model.coef_norm()
for k,v in glm_varimp.iteritems():
    glm_varimp[k] = abs(glm_varimp[k])
    
# Sort in descending order by magnitude
glm_sorted = sorted(glm_varimp.items(), key = operator.itemgetter(1), reverse = True)
table = tabulate.tabulate(glm_sorted, headers = ["Predictor", "Normalized Coefficient"], tablefmt = "orgtbl")
print "Variable Importances:\n\n" + table

# Deep Learning - Predict Delays
deeplearning_model = H2ODeepLearningEstimator(
    distribution = "bernoulli", model_id = "deeplearning_model_from_python",
    epochs = 100, hidden = [200,200],  
    seed = 6765686131094811000, variable_importances = True)
deeplearning_model.train(x               = myX,
                         y               = myY,
                         training_frame  = airlines_hex)

print "AUC of the training set : " + str(deeplearning_model.auc())
deeplearning_model.varimp(table)

h2o.shutdown(prompt=False)

