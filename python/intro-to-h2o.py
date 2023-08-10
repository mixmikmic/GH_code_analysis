# Load the H2O library and start up the H2O cluter locally on your machine
import h2o

# Number of threads, nthreads = -1, means use all cores on your machine
# max_mem_size is the maximum memory (in GB) to allocate to H2O
h2o.init(nthreads = -1, max_mem_size = 8)

loan_csv = "/Volumes/H2OTOUR/loan.csv"  # modify this for your machine
# Alternatively, you can import the data directly from a URL
#loan_csv = "https://raw.githubusercontent.com/h2oai/app-consumer-loan/master/data/loan.csv"
data = h2o.import_file(loan_csv)  # 163,987 rows x 15 columns

data.shape

data['bad_loan'] = data['bad_loan'].asfactor()  #encode the binary repsonse as a factor
data['bad_loan'].levels()  #optional: after encoding, this shows the two factor levels, '0' and '1'

# Partition data into 70%, 15%, 15% chunks
# Setting a seed will guarantee reproducibility

splits = data.split_frame(ratios=[0.7, 0.15], seed=1)  

train = splits[0]
valid = splits[1]
test = splits[2]

print train.nrow
print valid.nrow
print test.nrow

y = 'bad_loan'
x = list(data.columns)

x.remove(y)  #remove the response
x.remove('int_rate')  #remove the interest rate column because it's correlated with the outcome

# List of predictor columns
x

# Import H2O GLM:
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

# Initialize the GLM estimator:
# Similar to R's glm() and H2O's R GLM, H2O's GLM has the "family" argument

glm_fit1 = H2OGeneralizedLinearEstimator(family='binomial', model_id='glm_fit1')

glm_fit1.train(x=x, y=y, training_frame=train)

glm_fit2 = H2OGeneralizedLinearEstimator(family='binomial', model_id='glm_fit2', lambda_search=True)
glm_fit2.train(x=x, y=y, training_frame=train, validation_frame=valid)

glm_perf1 = glm_fit1.model_performance(test)
glm_perf2 = glm_fit2.model_performance(test)

# Print model performance
print glm_perf1
print glm_perf2

# Retreive test set AUC
print glm_perf1.auc()
print glm_perf2.auc()

# Compare test AUC to the training AUC and validation AUC
print glm_fit2.auc(train=True)
print glm_fit2.auc(valid=True)

# Import H2O RF:
from h2o.estimators.random_forest import H2ORandomForestEstimator

# Initialize the RF estimator:

rf_fit1 = H2ORandomForestEstimator(model_id='rf_fit1', seed=1)

rf_fit1.train(x=x, y=y, training_frame=train)

rf_fit2 = H2ORandomForestEstimator(model_id='rf_fit2', ntrees=100, seed=1)
rf_fit2.train(x=x, y=y, training_frame=train)

rf_perf1 = rf_fit1.model_performance(test)
rf_perf2 = rf_fit2.model_performance(test)

# Retreive test set AUC
print rf_perf1.auc()
print rf_perf2.auc()

rf_fit3 = H2ORandomForestEstimator(model_id='rf_fit3', seed=1, nfolds=5)
rf_fit3.train(x=x, y=y, training_frame=data)

print rf_fit3.auc(xval=True)

# Import H2O GBM:
from h2o.estimators.gbm import H2OGradientBoostingEstimator

# Initialize and train the GBM estimator:

gbm_fit1 = H2OGradientBoostingEstimator(model_id='gbm_fit1', seed=1)
gbm_fit1.train(x=x, y=y, training_frame=train)

gbm_fit2 = H2OGradientBoostingEstimator(model_id='gbm_fit2', ntrees=500, seed=1)
gbm_fit2.train(x=x, y=y, training_frame=train)

# Now let's use early stopping to find optimal ntrees

gbm_fit3 = H2OGradientBoostingEstimator(model_id='gbm_fit3', 
                                        ntrees=500, 
                                        score_tree_interval=5,     #used for early stopping
                                        stopping_rounds=3,         #used for early stopping
                                        stopping_metric='AUC',     #used for early stopping
                                        stopping_tolerance=0.0005, #used for early stopping
                                        seed=1)

# The use of a validation_frame is recommended with using early stopping
gbm_fit3.train(x=x, y=y, training_frame=train, validation_frame=valid)

gbm_perf1 = gbm_fit1.model_performance(test)
gbm_perf2 = gbm_fit2.model_performance(test)
gbm_perf3 = gbm_fit3.model_performance(test)

# Retreive test set AUC
print gbm_perf1.auc()
print gbm_perf2.auc()
print gbm_perf3.auc()

gbm_fit2.scoring_history()

gbm_fit3.scoring_history()

# Import H2O DL:
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

# Initialize and train the DL estimator:

dl_fit1 = H2ODeepLearningEstimator(model_id='dl_fit1', seed=1)
dl_fit1.train(x=x, y=y, training_frame=train)

dl_fit2 = H2ODeepLearningEstimator(model_id='dl_fit2', 
                                   epochs=20, 
                                   hidden=[10,10], 
                                   stopping_rounds=0,  #disable early stopping
                                   seed=1)
dl_fit2.train(x=x, y=y, training_frame=train)

dl_fit3 = H2ODeepLearningEstimator(model_id='dl_fit3', 
                                   epochs=20, 
                                   hidden=[10,10],
                                   score_interval=1,          #used for early stopping
                                   stopping_rounds=3,         #used for early stopping
                                   stopping_metric='AUC',     #used for early stopping
                                   stopping_tolerance=0.0005, #used for early stopping
                                   seed=1)
dl_fit3.train(x=x, y=y, training_frame=train, validation_frame=valid)

dl_perf1 = dl_fit1.model_performance(test)
dl_perf2 = dl_fit2.model_performance(test)
dl_perf3 = dl_fit3.model_performance(test)

# Retreive test set AUC
print dl_perf1.auc()
print dl_perf2.auc()
print dl_perf3.auc()

dl_fit3.scoring_history()

# Import H2O NB:
from h2o.estimators.naive_bayes import H2ONaiveBayesEstimator

# Initialize and train the NB estimator:

nb_fit1 = H2ONaiveBayesEstimator(model_id='nb_fit1')
nb_fit1.train(x=x, y=y, training_frame=train)

nb_fit2 = H2ONaiveBayesEstimator(model_id='nb_fit2', laplace=6)
nb_fit2.train(x=x, y=y, training_frame=train)

nb_perf1 = nb_fit1.model_performance(test)
nb_perf2 = nb_fit2.model_performance(test)

# Retreive test set AUC
print nb_perf1.auc()
print nb_perf2.auc()

h2o.shutdown(prompt=False)

