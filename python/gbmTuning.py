# Launch H2O Cluster
import h2o
h2o.init(nthreads = -1)

# Import data
df = h2o.import_file(path = "../data/titanic.csv")

# Convert some integer columns to factor/categorical
df["survived"] = df["survived"].asfactor()
df["ticket"] = df["ticket"].asfactor()

# Set predictors and response variable
response = "survived"

predictors = df.columns
predictors.remove("survived")
predictors.remove("name")

# Split the data for machine learning
splits = df.split_frame(ratios = [0.6, 0.2], 
                        destination_frames = ["train.hex", "valid.hex", "test.hex"],
                        seed = 1234)

train = splits[0]
valid = splits[1]
test = splits[2]

# Establish a Baseline

from h2o.estimators import H2OGeneralizedLinearEstimator
glm_model = H2OGeneralizedLinearEstimator(family = "binomial", 
                                          model_id = "glm_default.hex")
glm_model.train(x = predictors, y = response, training_frame = train, validation_frame = valid)

from h2o.estimators import H2ORandomForestEstimator
drf_model = H2ORandomForestEstimator(model_id = "drf_default.hex")
drf_model.train(x = predictors, y = response, training_frame = train, validation_frame = valid)

from h2o.estimators import H2OGradientBoostingEstimator
gbm_model = H2OGradientBoostingEstimator(model_id = "gbm_default.hex")
gbm_model.train(x = predictors, y = response, training_frame = train, validation_frame = valid)

from h2o.estimators import H2ODeepLearningEstimator
dl_model = H2ODeepLearningEstimator(model_id = "dl_default.hex")
dl_model.train(x = predictors, y = response, training_frame = train, validation_frame = valid)

header = ["Model", "Training AUC", "Validation AUC"]
table = [
    ["GLM", glm_model.auc(train = True), glm_model.auc(valid = True)],
    ["DRF", drf_model.auc(train = True), drf_model.auc(valid = True)],
    ["GBM", gbm_model.auc(train = True), gbm_model.auc(valid = True)],
    ["DL", dl_model.auc(train = True), dl_model.auc(valid = True)]
]
h2o.display.H2ODisplay(table, header)

# Investigate GBM Model
get_ipython().magic('matplotlib inline')
gbm_model.partial_plot(train, cols = ["sex", "age"], plot = True, plot_stddev = False)

# Decrease Learning Rate
gbm_learn_rate = H2OGradientBoostingEstimator(learn_rate = 0.05,
                                              model_id = "gbm_learnrate.hex")
gbm_learn_rate.train(x = predictors, y = response, training_frame = train, validation_frame = valid)

print "Learn Rate AUC: " + str(gbm_learn_rate.auc(valid = True))

# Use Early Stopping

# Early stopping once the moving average (window length = 5) of the validation AUC 
# doesn’t improve by at least 0.1% for 5 consecutive scoring events
    
gbm_early_stopping = H2OGradientBoostingEstimator(learn_rate = 0.05,
                                                  score_tree_interval = 10, 
                                                  stopping_rounds = 5, 
                                                  stopping_metric = "AUC", 
                                                  stopping_tolerance = 0.001,
                                                  ntrees = 5000,
                                                  model_id = "gbm_early_stopping.hex")
gbm_early_stopping.train(x = predictors, y = response, training_frame = train, validation_frame = valid)

print "Early Stopping AUC: " + str(gbm_early_stopping.auc(valid = True))

# Import H2O Grid Search
from h2o.grid.grid_search import H2OGridSearch

# Use Cartesian Grid Search to find best max depth
# Max depth can have a big impact on training time so we will first narrow down the best max depths

hyper_params = {'max_depth' : range(1, 25, 2)}

gbm_grid = H2OGradientBoostingEstimator(
    ## more trees is better if the learning rate is small enough 
    ## here, use "more than enough" trees - we have early stopping
    ntrees = 5000, 
    
    ## smaller learning rate is better
    ## since we have learning_rate_annealing, we can afford to start with a bigger learning rate
    learn_rate = 0.05,                                                         
    
    ## learning rate annealing: learning_rate shrinks by 1% after every tree 
    ## (use 1.00 to disable, but then lower the learning_rate)
    learn_rate_annealing = 0.99,                                               
    
    ## sample 80% of rows per tree
    sample_rate = 0.8,                                                       
   
    ## sample 80% of columns per split
    col_sample_rate = 0.8, 
    
    ## fix a random number generator seed for reproducibility
    seed = 1234,                                                             
    
    ## early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events
    stopping_rounds = 5,
    stopping_tolerance = 0.001,
    stopping_metric = "AUC", 
  
    ## score every 10 trees to make early stopping reproducible (it depends on the scoring interval)
    score_tree_interval = 10    
)

# Build grid search with previously made GBM and hyper parameters
grid = H2OGridSearch(gbm_grid, 
                     hyper_params, 
                     grid_id = 'depth_grid',
                     search_criteria = {'strategy': "Cartesian"})

# Train grid search
grid.train(x = predictors, 
           y = response,
           training_frame = train,
           validation_frame = valid)

## by default, display the grid search results sorted by increasing logloss (since this is a classification task)

grid   

## sort the grid models by decreasing AUC

sorted_grid = grid.get_grid(sort_by="auc", decreasing = True)    
sorted_grid.sorted_metric_table()[0:4]

## find the range of max_depth for the top 5 models
top_depths = sorted_grid.sorted_metric_table()['max_depth'][0:4] 
new_min = int(min(top_depths, key = int))
new_max = int(max(top_depths, key = int))

# Final Random Discrete Hyper-parameterization
import math

hyper_params_tune = {'max_depth': list(range(new_min, new_max + 1, 1)),
                     'sample_rate': [x/100. for x in range(20, 101)],
                     'col_sample_rate': [x/100. for x in range(20, 101)],
                     'min_rows': [2**x for x in range(0, int(math.log(train.nrow, 2) - 1) + 1)],
                     'nbins_cats': [2**x for x in range(4, 13, 1)],
                     'histogram_type': ["UniformAdaptive", "QuantilesGlobal"]
                    }

search_criteria_tune = {
    ## Random grid search
    'strategy': "RandomDiscrete",
    
    ## limit the runtime to 60 minutes
    'max_runtime_secs': 3600,         
  
    ## build no more than 100 models
    'max_models': 100,                  
  
    ## random number generator seed to make sampling of parameter combinations reproducible
    'seed': 1234,                        
  
    ## early stopping once the leaderboard of the top 5 models is converged to 0.1% relative difference
    'stopping_rounds': 5,                
    'stopping_metric': "AUC",
    'stopping_tolerance': 0.001
}

gbm_final_grid = H2OGradientBoostingEstimator(
    ## more trees is better if the learning rate is small enough 
    ## here, use "more than enough" trees - we have early stopping
    ntrees = 5000, 
    
    ## smaller learning rate is better
    ## since we have learning_rate_annealing, we can afford to start with a bigger learning rate
    learn_rate = 0.05,                                                         
    
    ## learning rate annealing: learning_rate shrinks by 1% after every tree 
    ## (use 1.00 to disable, but then lower the learning_rate)
    learn_rate_annealing = 0.99,      
    
    ## fix a random number generator seed for reproducibility
    seed = 1234,                                                             
    
    ## early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events
    stopping_rounds = 5,
    stopping_tolerance = 0.001,
    stopping_metric = "AUC", 
    
    ## early stopping based on timeout (no model should take more than 1 hour - modify as needed)
    max_runtime_secs = 3600,
  
    ## score every 10 trees to make early stopping reproducible (it depends on the scoring interval)
    score_tree_interval = 10    
)

final_grid = H2OGridSearch(gbm_final_grid, 
                           hyper_params = hyper_params_tune,
                           grid_id = 'final_grid',
                           search_criteria = search_criteria_tune)

# Train final grid search
final_grid.train(x = predictors, 
                 y = response,
                 training_frame = train,
                 validation_frame = valid)

## Sort the grid models by AUC

sorted_final_grid = final_grid.get_grid(sort_by = "auc", decreasing = True)    
sorted_final_grid.sorted_metric_table()[0:4]

# Final Test Scoring
# How well does our best model do on the final hold out dataset

best_model = h2o.get_model(sorted_final_grid.sorted_metric_table()['model_ids'][0])
performance_best_model = best_model.model_performance(test)

print "AUC on validation: " + str(best_model.auc(valid = True))
print "AUC on test: " + str(performance_best_model.auc())

# Shutdown h2o cluster
h2o.cluster().shutdown()



