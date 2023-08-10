import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='.*/jupyter/.*')

# Load the H2O library and start up the H2O cluter locally on your machine
import h2o
import numpy as np
import pandas as pd

# Number of threads, nthreads = -1, means use all cores on your machine
# max_mem_size is the maximum memory (in GB) to allocate to H2O
h2o.init(nthreads = -1)
h2o.cluster().show_status()
#remove deprecated warning messages

# A small clean telecommunications sample dataset (https://www.ibm.com/communities/analytics/watson-analytics-blog/predictive-insights-in-the-telco-customer-churn-data-set/
telco_dataset = h2o.import_file("https://s3.amazonaws.com/h2o-smalldata/TelcoChurn.csv")
# select all columns as predictors except the customerID (which is like an index) and the response column
features_list = list(telco_dataset.columns[1:-1])
response_name = 'Churn'
# specify the response column
response_col = telco_dataset['Churn']

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

# Partition data into 70%, 15%, 15% chunks
# Setting a seed will guarantee reproducibility
splits = telco_dataset.split_frame(ratios=[0.75,0.15], seed=1234)

train = splits[0]
valid = splits[1]
test = splits[2]

# Import H2O GBM:
from h2o.estimators.gbm import H2OGradientBoostingEstimator

# Initialize and train the GBM estimator:

gbm_fit1 = H2OGradientBoostingEstimator(model_id='gbm_fit1', seed=1234)
gbm_fit1.train(x=features_list, y=response_name, training_frame=train)

performance_train = gbm_fit1.model_performance(train)
print performance_train.auc()
performance_valid = gbm_fit1.model_performance(valid)
print performance_valid.auc()

models_predictions = gbm_fit1.predict(valid)
print models_predictions 

intervention_cost = 3.0  # Cost of classification 
effectiveness = 0.1      # 10% of users will be influenced by this particular intervention

# this is temporary for debubgging only
# we can set a threshold to use for now (this will be a variable in the future)
newdata = valid
model = gbm_fit1
threshold = 0.6
pred = model.predict(newdata)
pred['predict'] = pred['Yes']>threshold

pred

conf = model.confusion_matrix()
print conf
print type(conf)

conf_df = conf.table.as_data_frame()
print conf_df

TN = conf_df.ix[0,1]  #True Negative
FN = conf_df.ix[0,2]  #False Negative
FP = conf_df.ix[1,1]  #False Positive
TP = conf_df.ix[1,2]  #True Positive

unit_full_price = 100
unit_discount = 0.2
discount_effectiveness = 0.3
discounted_unit_price = (1 - unit_discount) * unit_full_price
print discounted_unit_price

# Total reward of TP group: TP * 80 * 0.3  # Discounted price is $80.00, which is 20% off of $100
TP_value = TP * discounted_unit_price * discount_effectiveness 
print TP_value

# Total cost of FP group: FP * 100 * .2  # Coupon is $20, which is 20% of $100, lost for each FP
FP_value = FP * unit_full_price * unit_discount * -1
print FP_value

# Total cost of FN group: 30% of these churns could have been saved at a 20% discount
# this is the amount of extra money you could have gained if you had predicted correctly that these people would churn
# note revove FN_value if you want to assume this is money you can't gain because your model is as good as it can be.
FN_value = FN * discounted_unit_price * discount_effectiveness * -1
print FN_value

# Total cost/reward of TN group: Nothing, the intervention has no effect on the outcome for this population
TN_value = 0.00
print TN_value 

intervention_net_value = TP_value + FP_value + TN_value + FN_value
print(TP_value + FP_value + TN_value + FN_value)
print intervention_net_value

print(TP_value)
print(FP_value)
print(TN_value)
print(FN_value)
TP_value + FP_value + TN_value + FN_value

def intervention_value(intervention_params, newdata, model, threshold = None):
    ''' 
    intervention_params is a dict specifying intervention parameters
    model must be a binomial H2O model
    threshold is a number between 0 and 1
    newdata is an H2OFrame of test data
    '''
    
    # Parse parameters
    unit_full_price = intervention_params['unit_full_price']
    unit_discount = intervention_params['unit_discount']
    discount_effectiveness = intervention_params['discount_effectiveness']
    discounted_unit_price = (1 - unit_discount) * unit_full_price #CHANGED THIS FROM unit_cost
    
#     assert(unit_discount > 0.0)
    
    if threshold is not None:
        # Update the predictions using specified threshold
        pred = model.predict(newdata)
        pred['predict'] = pred['Yes']>threshold
        
    print(threshold)
    
    # Confusion matrix
    conf = model.confusion_matrix()
    print conf
    conf_df = conf.table.as_data_frame()
    TN = conf_df.ix[0,1]  #True Negative
    FN = conf_df.ix[0,2]  #False Negative
    FP = conf_df.ix[1,1]  #False Positive
    TP = conf_df.ix[1,2]  #True Positive
    
    # Total reward of TP group: TP * 0.6 * 0.3  #Discounted price is $0.60, which is 40% off of $1.00
    TP_value = TP * discounted_unit_price * discount_effectiveness 
    print TP_value

    # Total cost of FP group: FP * 0.4 * 1.00  #Coupon is $0.40, which is 40% of $1.00, lost for each FP
    FP_value = FP * unit_full_price * unit_discount * -1
    print FP_value
    
    # Total cost of FN group: 30% of these churns could have been saved at a 40% discount
    FN_value = FN * discounted_unit_price * discount_effectiveness * -1
    print("False Negative", FN_value)

    # Total cost/reward of TN group: Nothing, the intervention has no effect on the outcome for this population
    TN_value = 0.00
    print TN_value 
    
    intervention_net_value = TP_value + FP_value + TN_value + FN_value
    print 'Value of intervention is %.2f'%intervention_net_value
    return intervention_net_value
    

# same as above but without false negatives
def intervention_value(intervention_params, newdata, model, threshold = None):
    ''' 
    intervention_params is a dict specifying intervention parameters
    model must be a binomial H2O model
    threshold is a number between 0 and 1
    newdata is an H2OFrame of test data
    '''
    
    # Parse parameters
    unit_full_price = intervention_params['unit_full_price']
    unit_discount = intervention_params['unit_discount']
    discount_effectiveness = intervention_params['discount_effectiveness']
    discounted_unit_price = (1 - unit_discount) * unit_full_price #CHANGED THIS FROM unit_cost
    
#     assert(unit_discount > 0.0)
    
    if threshold is not None:
        # Update the predictions using specified threshold
        pred = model.predict(newdata)
        pred['predict'] = pred['Yes']>threshold
    
    # Confusion matrix
    conf = model.confusion_matrix()
    print conf
    conf_df = conf.table.as_data_frame()
    TN = conf_df.ix[0,1]  #True Negative
    FN = conf_df.ix[0,2]  #False Negative
    FP = conf_df.ix[1,1]  #False Positive
    TP = conf_df.ix[1,2]  #True Positive
    
    # Total reward of TP group: TP * 0.6 * 0.3  #Discounted price is $0.60, which is 40% off of $1.00
    TP_value = TP * discounted_unit_price * discount_effectiveness 
    print TP_value

    # Total cost of FP group: FP * 0.4 * 1.00  #Coupon is $0.40, which is 40% of $1.00, lost for each FP
    FP_value = FP * unit_full_price * unit_discount * -1
    print FP_value
    
    # Total cost of FN group: 30% of these churns could have been saved at a 40% discount
    FN_value = FN * discounted_unit_price * discount_effectiveness * -1
    print FN_value

    # Total cost/reward of TN group: Nothing, the intervention has no effect on the outcome for this population
    TN_value = 0.00
    print TN_value 
    
    intervention_net_value = TP_value + FP_value + TN_value # + FN_value
    print 'Value of intervention is %.2f'%intervention_net_value
    return intervention_net_value
    

# Let's try some interventions:


intervention1 = {'unit_full_price': 100,
                 'unit_discount': 0.2,
                 'discount_effectiveness': .3}

ival1 = intervention_value(intervention_params = intervention1, newdata = test, model = model, threshold = None)

# A unit discount of $0.00 should produce a value of intervention of 0.00

intervention2 = {'unit_full_price': 100,
                 'unit_discount': 0.2,
                 'discount_effectiveness': 0.3}

ival2 = intervention_value(intervention_params = intervention2, newdata = test, model = model, threshold = None)

# A unit discount of $0.00 should produce a value of intervention of 0.00... hmm maybe something is wrong above?

intervention2 = {'unit_full_price': 100,
                 'unit_discount': 0.0,
                 'discount_effectiveness': 0.3}

ival = intervention_value(intervention_params = intervention2, newdata = test, model = model, threshold = None)

# load the best model from the churn_analysis notebook
best_gbm = h2o.load_model("/Users/laurend/Code/repos/customer-churn/data/")

import numpy as np
# define the input range of values for the cost function to take:
# $20 to $180, 0 - 100%, and fixed number of people
# using verizon current range of plans as a values for possible monthly cell phone bills
range_unit_full_price = [35, 50, 70, 90, 110]
# discount can be from 0 - 100% off using 10% increments
range_unit_discount = np.arange(0.0, 1.1, .10)
# effectiveness of coupon can be 0 to 100% effective, using increments of 10%
range_discount_effectiveness = np.arange(0.0, 1.1, .10)
# provide a range of relative thresholds 
# range_threshold = np.arange(.10, 1.1, .10)

# calculate the discout_unit_price given a unit_discount and unit full price values
# TO DO: make a discount_unit_price definition:
# discounted_unit_price = (1 - unit_discount) * unit_full_price
# print discounted_unit_price

# same as definition above but requred intervention_params to be individual inputs
# the main difference is that FN_value is not used, because we assume this is a margin lost regardless of your
# actions
def intervention_gains(price, discount_amount, discount_effectiveness, newdata, model, threshold = None):
    ''' 
    price: is the unit price
    discount_amount: is the unit discount
    discount_effectiveness: is the discount effectiveness percent
    model must be a binomial H2O model
    threshold is a number between 0 and 1
    newdata is an H2OFrame of test data
    returns: the gains or loss from type of intervention
    '''
    
#     assert(unit_discount > 0.0)

    unit_full_price = price
    unit_discount = discount_amount
    discount_effectiveness = discount_effectiveness
    discounted_unit_price = (1 - unit_discount) * unit_full_price #CHANGED THIS FROM unit_cost
    
    if threshold is not None:
        # Update the predictions using specified threshold
        pred = model.predict(newdata)
        pred['predict'] = pred['Yes']>threshold
    
    # Confusion matrix
    conf = model.confusion_matrix()
    conf_df = conf.table.as_data_frame()
    TN = conf_df.ix[0,1]  #True Negative
#     FN = conf_df.ix[0,2]  #False Negative
    FP = conf_df.ix[1,1]  #False Positive
    TP = conf_df.ix[1,2]  #True Positive
    
    # Total reward of TP group: TP * 0.6 * 0.3  #Discounted price is $0.60, which is 40% off of $1.00
    TP_value = TP * discounted_unit_price * discount_effectiveness 

    # Total cost of FP group: FP * 0.4 * 1.00  #Coupon is $0.40, which is 40% of $1.00, lost for each FP
    FP_value = FP * unit_full_price * unit_discount * -1
    
    # Total cost of FN group: 30% of these churns could have been saved at a 40% discount
#     FN_value = TN * discounted_unit_price * discount_effectiveness * -1

    # Total cost/reward of TN group: Nothing, the intervention has no effect on the outcome for this population
    TN_value = 0.00
    
    intervention_net_value = TP_value + FP_value + TN_value # + FN_value
    return intervention_net_value
    

# TO DO: change model to 'best_model' once you have loaded best model from the saved binary
# TO DO: save a glm and see how it compares to the tuned GBM
# 1) create dictionary with the range of of values to loop over
# 2) create an accuracy threshold look up table, that returns the threshold based on an input accuracy value


# 3) loop over the monthly_bill (aka unit_full_price), discounts (aka unit_discount), 
#    effectiveness (aka discount_effectiveness), accuracy is related to the threshold
#    maybe instead of accuracy this could be confidence and allows someone to change the threshold
#    and build another dictionary that contains price, discount, effectiveness, accuracy, intervention_value
#    convert the final dictionary to a pandas dataframe that can then be used as a lookup table and also converted
#    to multiple pivot tables

intervention_dict = {'Monthly_bill': [35, 50, 70, 90, 110],
                     'Discounts': np.arange(0.0, 1.1, .10),
                     'Discount_Effectiveness': np.arange(0.0, 1.1, .10),
                     'Confidence_threshold': np.arange(.10, 1.1, .10)
    
}

# use best_gbm as the model when you calculate gains (or losses)
# initialized a list to fill with dict with new values (don't use confidence for the first naive run)
temp_dict = []

# loop over the intervention dict
for bill in intervention_dict['Monthly_bill']:
    for discount in intervention_dict['Discounts']:
        for effectiveness in intervention_dict['Discount_Effectiveness']:
            
            # calculate the intervention gains (or losses)
            gains_loss = intervention_gains(price=bill, discount_amount=discount, 
                                        discount_effectiveness=effectiveness, 
                                        newdata = test, model = model, threshold = None)
            
            # append dictionary results to list
            temp_dict.append({'Monthly_bill': bill, 'Discounts': discount,
                  'Discount_Effectiveness': effectiveness, 'Confidence_threshold': None, 'Gains/Loss': gains_loss})
            
            
            
# convert to pandas dataframe when for loops are finished, reset order of the columns
lookup_table = pd.DataFrame(temp_dict, columns = ['Monthly_bill', 'Discounts', 'Discount_Effectiveness',
                                   'Confidence_threshold' ,'Gains/Loss'])
lookup_table.head()

# lookup_table.to_csv('/Users/laurend/Code/repos/customer-churn/data/lookup_table.csv')

# for testing purposes:
# intervention_dict = {'Monthly_bill': [35, 50, 70, 90, 110],
#                      'Discounts': np.arange(0.0, 1.1, .10),
#                      'Discount_Effectiveness': np.arange(0.0, 1.1, .10),
#                      'Confidence_threshold': np.arange(.10, 1.1, .10)}
# gains_loss = intervention_gains(price=bill, discount_amount=discount, 
#                                discount_effectiveness=effectiveness, newdata = test, model = model, threshold = None)

# definition to create a lookup table

# TO DO: turn this into a definition and apply to glm
# 1) create dictionary with the range of of values to loop over
# 2) create an accuracy threshold look up table, that returns the threshold based on an input accuracy value


# 3) loop over the monthly_bill (aka unit_full_price), discounts (aka unit_discount), 
#    effectiveness (aka discount_effectiveness), accuracy is related to the threshold
#    maybe instead of accuracy this could be confidence and allows someone to change the threshold
#    and build another dictionary that contains price, discount, effectiveness, accuracy, intervention_value
#    convert the final dictionary to a pandas dataframe that can then be used as a lookup table and also converted
#    to multiple pivot tables

intervention_dict = {'Monthly_bill': [35, 50, 70, 90, 110],
                     'Discounts': np.arange(0.0, 1.1, .10),
                     'Discount_Effectiveness': np.arange(0.0, 1.1, .10),
                     'Confidence_threshold': np.arange(.10, 1.1, .10)
    
}

def gains_loss_table(intervention_params, model_used, data_used, effectiveness_used, 
                    discount_used, price_used, threshold_used=None):
    # use best_gbm as the model when you calculate gains (or losses)
    # initialized a list to fill with dict with new values (don't use confidence for the first naive run)
    temp_dict = []

    # loop over the intervention dict
    for bill in intervention_dict['Monthly_bill']:
        for discount in intervention_dict['Discounts']:
            for effectiveness in intervention_dict['Discount_Effectiveness']:

                # calculate the intervention gains (or losses)
                gains_loss = intervention_gains(price=price_used, discount_amount=discount_used, 
                                            discount_effectiveness=effectiveness_used, 
                                            newdata = data_used, model = model_used, threshold = threshold_used)

                # append dictionary results to list
                temp_dict.append({'Monthly_bill': bill, 'Discounts': discount,
                      'Discount_Effectiveness': effectiveness, 'Confidence_threshold': None, 'Gains/Loss': gains_loss})



    # convert to pandas dataframe when for loops are finished, reset order of the columns
    lookup_table = pd.DataFrame(temp_dict, columns = ['Monthly_bill', 'Discounts', 'Discount_Effectiveness',
                                       'Confidence_threshold' ,'Gains/Loss'])
    
    return lookup_table

# get the same results but for a baseline glm
telco_dataset = h2o.import_file("https://s3.amazonaws.com/h2o-smalldata/TelcoChurn.csv")
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
features_list = list(telco_dataset.columns[1:-1])
response_name = 'Churn'
train, valid, test = telco_dataset.split_frame(ratios=[0.70,0.15], seed=1234)
glm = H2OGeneralizedLinearEstimator(family="binomial")
glm.train(x=features_list, y=response_name, training_frame=train,validation_frame = valid)

# gains_loss_table()

print(glm.auc(valid=True))



