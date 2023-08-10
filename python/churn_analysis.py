# h2o.cluster().shutdown()

import h2o
import numpy as np
import math
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='.*/jupyter/.*')
h2o.init(nthreads = -1)

# A small clean telecommunications sample dataset (https://www.ibm.com/communities/analytics/watson-analytics-blog/predictive-insights-in-the-telco-customer-churn-data-set/
telco_dataset = h2o.import_file("https://s3.amazonaws.com/h2o-smalldata/TelcoChurn.csv")

# update the telco data format: 
# change SeniorCitizen to 'yes'/ 'no'
telco_dataset['SeniorCitizen'] = (telco_dataset['SeniorCitizen'] == 1).ifelse('Yes','No')
# Add the same form of capitalization across variables
# columns changed were: customerID, gender, tenure
telco_dataset.columns =[u'CustomerID',
 u'Gender',
 u'SeniorCitizen',
 u'Partner',
 u'Dependents',
 u'Tenure',
 u'PhoneService',
 u'MultipleLines',
 u'InternetService',
 u'OnlineSecurity',
 u'OnlineBackup',
 u'DeviceProtection',
 u'TechSupport',
 u'StreamingTV',
 u'StreamingMovies',
 u'Contract',
 u'PaperlessBilling',
 u'PaymentMethod',
 u'MonthlyCharges',
 u'TotalCharges',
 u'Churn']

# get a summary of the dataset
print telco_dataset.nacnt()
telco_dataset.describe()

# what does each column tell you, make sure type is correct, and see if new features need to be created with strings
telco_dataset.columns

# check whether the customerID column is unique per row, if so use as an index and remove from the predictors
telco_dataset['CustomerID'].asfactor().unique().nrow

# select all columns as predictors except the customerID (which is like an index) and the response column
features_list = list(telco_dataset.columns[1:-1])
response_name = 'Churn'

# specify the response column
response_col = telco_dataset[response_name]
# get a list of the categorical levels in your response column
print 'the response classes are:',response_col.levels()
print 'number of classes:', response_col.nlevels()
print ''
# check that the response column is already interpreted as a factor (i.e. enum/categorical)
print 'Is the response column a categorical:',response_col.isfactor()
# check that there are two levels in our response column:
response_col.nlevels()
print ''
# check for missing values in the training set and specifically the response column
print 'there are {0} missing values in the dataset'.format(telco_dataset.isna().sum())
print 'there are {0} missing values in the labels'.format(telco_dataset[response_col].isna().sum())
print ''

print 'check for class imbalace'
print '------------------------'
num_train_samples = telco_dataset.shape[0]  # Total number of training samples
print 'the dataset is not imbalanced, neither class is less then 10% of the whole (as shown below)'
telco_dataset[response_name].table()['Count']/num_train_samples


# create a new column that is the same as tenure but it replaces tenure = 0 with tenure = 1
# tenure = 0 corresponds to people who have be a customer less then a month but have still payed
# the monthly fee:
telco_dataset['new_tenure_col'] = (telco_dataset['Tenure'] == 0).ifelse(1, telco_dataset['Tenure'])

# impute TotalCharges from MonthlyCharges and tenure (i.e. TotalCharges = tenure * MonthlyCharges)
# if TotalCharges value is missing fill in, if not leave as is
telco_dataset['TotalCharges'] = (telco_dataset['TotalCharges'].isna() == 1).ifelse(
    (telco_dataset['MonthlyCharges']  * telco_dataset['new_tenure_col']), telco_dataset['TotalCharges'])

# remove new tenure col when done
telco_dataset= telco_dataset.drop('new_tenure_col')
# check that there are no more missing values
print telco_dataset['TotalCharges'].isna().sum()

# print out the count for each categorical feature level (this excludes monthly charges and total charges)
for column_name in telco_dataset.columns[1:-3]:
    print telco_dataset[str(column_name)].table()
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    

# split your datasets and set seed so the split is the same each time the code is run
train, valid, test = telco_dataset.split_frame(ratios=[0.70,0.15], seed=1234)

# check that the split went as expected, print out the dimensions of each dataset
# and then print their sum
print train.shape
print valid.shape
print test.shape
print train.shape[0] + valid.shape[0] + test.shape[0]

# run gbm estimator with the default parameters to establish a benchmark model 

# import GBM estimator with default parameters (set seed for reproducibility)
from h2o.estimators.gbm import H2OGradientBoostingEstimator
default_model = H2OGradientBoostingEstimator(distribution= 'bernoulli', seed = 1234)

# train the default model
default_model.train(x=features_list, y=response_name, training_frame=train)

# get the AUC for the training set
print 'train auc:', default_model.auc() 

# get the AUC for the validation set
default_perf_on_valid = default_model.model_performance(valid)
print 'validation auc:', default_perf_on_valid.auc()

# using parameters from https://github.com/h2oai/h2o-3/blob/master/h2o-docs/src/product/tutorials/gbm/gbmTuning.ipynb
# this model overfits less on the training set
model_0 = H2OGradientBoostingEstimator(distribution='bernoulli',
                                    ntrees=10000,
                                    max_depth=4,
                                    learn_rate=0.01,
                                    stopping_rounds = 5, stopping_tolerance = 1e-4, stopping_metric = "AUC",
                                    sample_rate = 0.8,
                                    col_sample_rate = 0.8,
                                    seed = 1234,
                                    score_tree_interval = 10)

model_0.train(x=features_list, 
              y=response_name, 
              training_frame=train,
              validation_frame = valid)

# get the AUC for the training set
print 'train auc:', model_0.auc() 

# get the AUC for the validation set
print 'validation auc:', model_0.auc(valid= True) 

# using grid to get the best max_depths
hyper_params = {'max_depth' : range(1,30,2)}
search_criteria = {'strategy': "Cartesian"}

# build gbm model with grid search parameters
from h2o.grid.grid_search import H2OGridSearch
gs_1 = H2OGridSearch(H2OGradientBoostingEstimator(distribution='bernoulli',
                                    ntrees=10000,
                                    learn_rate=0.01,
                                    # learn_rate_annealing = 0.99, 
                                    sample_rate = 0.8,
                                    col_sample_rate = 0.8,
                                    seed = 1234,
                                    score_tree_interval = 10,              
                                    stopping_rounds = 5,
                                    stopping_metric = "AUC",
                                    stopping_tolerance = 1e-4),
                                    hyper_params = hyper_params,
                                    grid_id = 'grid_determines_max_depth',
                                    search_criteria = search_criteria)

# train grid search
gs_1.train(x=features_list, 
           y=response_name, 
           training_frame=train,
           validation_frame = valid)

# get the grid search results to see which max_depth performed the best
print(gs_1)

# print out the auc for all models, sorted from best to worst
auc_table = gs_1.sort_by('auc(valid=True)',increasing=False)
print(auc_table)

# this is breaking because of xrange(), hard coding results for now
# # find the range of the max_depth for the top five models
# new_auc_table = auc_table[1:5]
# max_depths_to_use = new_auc_table['Hyperparameters: [max_depth]']
# print max_depths_to_use

# # get the max depths as a list
# new_maxmin_list = []
# for element in max_depths_to_use:
#     new_maxmin_list.append(element[0])
# new_max = max(new_maxmin_list)
# new_min = min(new_maxmin_list)
new_min = 1
new_max = 9


hyper_params_2 = {'max_depth' : list(range(new_min,new_max+1,1)),
                'sample_rate': [x/100. for x in range(20,101)],
                'col_sample_rate' : [x/100. for x in range(20,101)],
                'col_sample_rate_per_tree': [x/100. for x in range(20,101)],
                'col_sample_rate_change_per_level': [x/100. for x in range(90,111)],
                'min_rows': [2**x for x in range(0,int(math.log(train.nrow,2)-1)+1)],
                'nbins': [2**x for x in range(4,11)],
                'nbins_cats': [2**x for x in range(4,13)],
                'min_split_improvement': [0,1e-8,1e-6,1e-4],
                'histogram_type': ["UniformAdaptive","QuantilesGlobal","RoundRobin"]}
search_criteria_2 = {'strategy': "RandomDiscrete",
                   'max_runtime_secs': 3600,  ## limit the runtime to 60 minutes
                   'max_models': 100,  ## build no more than 100 models
                   'seed' : 1234,
                   'stopping_rounds' : 5,
                   'stopping_metric' : "AUC",
                   'stopping_tolerance': 1e-3
                   }

# printing out values in hyperparams_2 to give a sense of the range of values
for element in hyper_params_2.keys():
    print element
    print hyper_params_2[element]
    print "------------------------------"


#Build grid search with GBM and hyper parameters
gs_2 = H2OGridSearch(H2OGradientBoostingEstimator(distribution='bernoulli',
                                    ntrees=10000,
                                    learn_rate=0.05,
                                    #learn_rate_annealing = 0.99,
                                    stopping_rounds = 5,
                                    stopping_tolerance = 1e-4,
                                    stopping_metric = "AUC", 
                                    score_tree_interval = 10,
                                    seed = 1234),
                                    hyper_params = hyper_params_2,
                                    grid_id = 'grid_2',
                                    search_criteria = search_criteria_2)

# train the grid and print results
gs_2.train(x=features_list, 
           y=response_name, 
           training_frame=train,
           validation_frame = valid)
print(gs_2)

# print out the auc for all of the models on the validation set
auc_table_2 = gs_2.sort_by('auc(valid=True)',increasing=False)
print(auc_table_2)

# get the best model from the list (the model name listed at the top of the table)
best_model = h2o.get_model('grid_2_model_9')
test_performance_model = best_model.model_performance(test)

# get the performance on the test model
print test_performance_model.auc()

# save a csv of the predictions:
# first get the predictions from the best model
best_model_predictions = best_model.predict(test)
best_model_predictions.head()

# export predictions as a csv to the current directory
# h2o.export_file(best_model_predictions,'best_model_predictions.csv')

# save model as a binary file to be uploaded and used in the cost function notebook
# h2o.save_model(best_model, path="/Users/laurend/Code/repos/customer-churn/data/")

# download a POJO of the best model (best_model), you have to specify the path to 
# where you want your pojo saved (not provided below)
# otherwise it will print to screen
# h2o.download_pojo(best_model)



