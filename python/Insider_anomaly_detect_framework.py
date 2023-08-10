import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

user_pc = pd.read_csv('./Data_Subset/Input_features/user_pc_gdegree.csv')

user_pc.head()

user_pc_ct = np.array(user_pc['pc_count'])

user_pc_ct

user_pc_ct = user_pc_ct.reshape(-1,1) # need to reshape(-1,1) for single feature, (1,-1) 
                                    #for single sample or multiple features
forest = IsolationForest()

forest.fit(user_pc_ct)

# Getting the anomaly score for each user

graph_a_score = forest.decision_function(user_pc_ct)
graph_a_score

type(graph_a_score)

graph_a_score.tolist()

# Define an empty datframe to store the result for the graph input features
graph_result = pd.DataFrame()

#graph_result['user'] = user_pc['user']
graph_result['ascore'] = graph_a_score

graph_result.loc[graph_result['ascore'] < 0] # possible outliers

# Save the graph result
graph_result.to_csv('./Data_Subset/IFResult/graph_result.csv', index = False)

# Load Logon/Logoff statistics data saved previously

user_logon_stats = pd.read_csv('./Data_Subset/Input_features/user_logon_stats.csv')

user_logon_stats.head()

user_logon_stats['min'].dtype

pd.to_datetime(user_logon_stats['min'][1]).hour

# Function to convert datetime 'time' to time in seconds

def dtt2timestamp(dtt):
    ts = (dtt.hour * 60 + dtt.minute) * 60 + dtt.second
    #if you want microseconds as well
    #ts += dtt.microsecond * 10**(-6)
    return ts

user_logon_stats1 = user_logon_stats

user_logon_stats1['min_dt'] = pd.to_datetime(user_logon_stats['min'])
user_logon_stats1['max_dt'] = pd.to_datetime(user_logon_stats['max'])
user_logon_stats1['mode_dt'] = pd.to_datetime(user_logon_stats['mode'])
user_logon_stats1['mean_dt'] = pd.to_datetime(user_logon_stats['mean'])

user_logon_stats1.head()

# use the function to generate time in sec value

min_ts = [dtt2timestamp(dtt) for dtt in user_logon_stats1['min_dt']] 
max_ts = [dtt2timestamp(dtt) for dtt in user_logon_stats1['max_dt']]
mode_ts = [dtt2timestamp(dtt) for dtt in user_logon_stats1['mode_dt']]
mean_ts = [dtt2timestamp(dtt) for dtt in user_logon_stats1['mean_dt']]


user_logon_stats1['min_ts'] = min_ts
user_logon_stats1['max_ts'] = max_ts
user_logon_stats1['mode_ts'] = mode_ts
user_logon_stats1['mean_ts'] = mean_ts

# new df to store the tsec values
user_logon_stats_tsec = pd.DataFrame()

user_logon_stats_tsec['user'] = user_logon_stats1['user']
user_logon_stats_tsec['min_ts'] = user_logon_stats1['min_ts']
user_logon_stats_tsec['max_ts'] = user_logon_stats1['max_ts']
user_logon_stats_tsec['mode_ts'] = user_logon_stats1['mode_ts']
user_logon_stats_tsec['mean_ts'] = user_logon_stats1['mean_ts']

user_logon_stats_tsec.head()

# save the data
user_logon_stats_tsec.to_csv('./Data_Subset/Input_features/user_logon_stats_tsec.csv', index = False)

user_logoff_stats = pd.read_csv('./Data_Subset/Input_features/user_logoff_stats.csv')

user_logoff_stats.head()

user_logoff_stats1 = user_logoff_stats

user_logoff_stats1['min_dt'] = pd.to_datetime(user_logoff_stats['min'])
user_logoff_stats1['max_dt'] = pd.to_datetime(user_logoff_stats['max'])
user_logoff_stats1['mode_dt'] = pd.to_datetime(user_logoff_stats['mode'])
user_logoff_stats1['mean_dt'] = pd.to_datetime(user_logoff_stats['mean'])

user_logoff_stats1.head()

# use the function to generate time in sec value

off_min_ts = [dtt2timestamp(dtt) for dtt in user_logoff_stats1['min_dt']] 
off_max_ts = [dtt2timestamp(dtt) for dtt in user_logoff_stats1['max_dt']]
off_mode_ts = [dtt2timestamp(dtt) for dtt in user_logoff_stats1['mode_dt']]
off_mean_ts = [dtt2timestamp(dtt) for dtt in user_logoff_stats1['mean_dt']]


user_logoff_stats1['min_ts'] =off_min_ts
user_logoff_stats1['max_ts'] = off_max_ts
user_logoff_stats1['mode_ts'] = off_mode_ts
user_logoff_stats1['mean_ts'] = off_mean_ts

user_logoff_stats1.head()

# new df to store the tsec values
user_logoff_stats_tsec = pd.DataFrame()

user_logoff_stats_tsec['user'] = user_logoff_stats1['user']
user_logoff_stats_tsec['min_ts'] = user_logoff_stats1['min_ts']
user_logoff_stats_tsec['max_ts'] = user_logoff_stats1['max_ts']
user_logoff_stats_tsec['mode_ts'] = user_logoff_stats1['mode_ts']
user_logoff_stats_tsec['mean_ts'] = user_logoff_stats1['mean_ts']

user_logoff_stats_tsec.head()

# save the data
user_logoff_stats_tsec.to_csv('./Data_Subset/Input_features/user_logoff_stats_tsec.csv', index = False)

# combined logon/logoff data for IForest input

ulog_on_off_stats = pd.DataFrame()

ulog_on_off_stats['user'] = user_logon_stats_tsec['user']

ulog_on_off_stats['on_min_ts'] = user_logon_stats_tsec['min_ts']
ulog_on_off_stats['on_max_ts'] = user_logon_stats_tsec['max_ts']
ulog_on_off_stats['on_mode_ts'] = user_logon_stats_tsec['mode_ts']
ulog_on_off_stats['on_mean_ts'] = user_logon_stats_tsec['mean_ts']

ulog_on_off_stats['off_min_ts'] = user_logoff_stats_tsec['min_ts']
ulog_on_off_stats['off_max_ts'] = user_logoff_stats_tsec['max_ts']
ulog_on_off_stats['off_mode_ts'] = user_logoff_stats_tsec['mode_ts']
ulog_on_off_stats['off_mean_ts'] = user_logoff_stats_tsec['mean_ts']

ulog_on_off_stats.head()

# save the data
ulog_on_off_stats.to_csv('./Data_Subset/Input_features/ulog_on_off_stats.csv', index = False)

# fit the model

# input array
ulog_on_off_stats.columns[1:]

ulog_stats = ulog_on_off_stats.as_matrix(columns=ulog_on_off_stats.columns[1:])
ulog_stats

# fit the model contd..

ulog_stats = ulog_stats #.reshape(1,-1) # need to reshape(-1,1) for single feature, (1,-1) 
                                      #for single sample or multiple features
forest = IsolationForest()

forest.fit(ulog_stats)

# anomaly score

ulog_ascore = forest.decision_function(ulog_stats)
ulog_ascore

ulog_ascore.shape

# Save the result
user_log_result = pd.DataFrame()

user_log_result['user'] = ulog_on_off_stats['user']

user_log_result['ascore'] = ulog_ascore

user_log_result.head()

# save the result

user_log_result.to_csv('./Data_Subset/IFResult/user_log_result.csv', index = False)

# Load the data

device_conn_stats = pd.read_csv('./Data_Subset/Input_features/device_conn_stats.csv')
device_disconn_stats = pd.read_csv('./Data_Subset/Input_features/device_disconn_stats.csv')
files_per_day_stats = pd.read_csv('./Data_Subset/Input_features/files_per_day_stats.csv')

device_conn_stats.head()

device_conn_stats1 = device_conn_stats

device_conn_stats1['min_dt'] = pd.to_datetime(device_conn_stats['min'])
device_conn_stats1['max_dt'] = pd.to_datetime(device_conn_stats['max'])
device_conn_stats1['mode_dt'] = pd.to_datetime(device_conn_stats['mode'])
device_conn_stats1['mean_dt'] = pd.to_datetime(device_conn_stats['mean'])

# use the function to generate time in sec value

con_min_ts = [dtt2timestamp(dtt) for dtt in device_conn_stats1['min_dt']]
con_max_ts = [dtt2timestamp(dtt) for dtt in device_conn_stats1['max_dt']]
con_mode_ts = [dtt2timestamp(dtt) for dtt in device_conn_stats1['mode_dt']]
con_mean_ts = [dtt2timestamp(dtt) for dtt in device_conn_stats1['mean_dt']]


device_conn_stats1.head()

# new dataframe 

device_conn_stats_tsec = pd.DataFrame()

device_conn_stats_tsec['user'] = device_conn_stats1['user']
device_conn_stats_tsec['con_min_ts'] = con_min_ts
device_conn_stats_tsec['con_max_ts'] = con_max_ts
device_conn_stats_tsec['con_mode_ts'] = con_mode_ts
device_conn_stats_tsec['con_mean_ts'] = con_mean_ts

device_conn_stats_tsec.head()

# save the data
device_conn_stats_tsec.to_csv('./Data_Subset/Input_features/device_conn_stats_tsec.csv', index = False)

device_disconn_stats1 = device_disconn_stats

device_disconn_stats1['min_dt'] = pd.to_datetime(device_disconn_stats['min'])
device_disconn_stats1['max_dt'] = pd.to_datetime(device_disconn_stats['max'])
device_disconn_stats1['mode_dt'] = pd.to_datetime(device_disconn_stats['mode'])
device_disconn_stats1['mean_dt'] = pd.to_datetime(device_disconn_stats['mean'])

device_disconn_stats1.head()

# use the function to generate time in sec value

dcon_min_ts = [dtt2timestamp(dtt) for dtt in device_disconn_stats1['min_dt']]
dcon_max_ts = [dtt2timestamp(dtt) for dtt in device_disconn_stats1['max_dt']]
dcon_mode_ts = [dtt2timestamp(dtt) for dtt in device_disconn_stats1['mode_dt']]
dcon_mean_ts = [dtt2timestamp(dtt) for dtt in device_disconn_stats1['mean_dt']]

# new dataframe 

device_disconn_stats_tsec = pd.DataFrame()

device_disconn_stats_tsec['user'] = device_disconn_stats1['user']
device_disconn_stats_tsec['dcon_min_ts'] = dcon_min_ts
device_disconn_stats_tsec['dcon_max_ts'] = dcon_max_ts
device_disconn_stats_tsec['dcon_mode_ts'] = dcon_mode_ts
device_disconn_stats_tsec['dcon_mean_ts'] = dcon_mean_ts

device_disconn_stats_tsec.head()

# save the data

device_disconn_stats_tsec.to_csv('./Data_Subset/Input_features/device_disconn_stats_tsec.csv', index = False)

files_per_day_stats.head()

# Combine all the removable media (device) parameters

device_stats = pd.DataFrame() # new df

device_stats['user'] = device_conn_stats_tsec['user']

# connect stats
device_stats['con_min_ts'] = device_conn_stats_tsec['con_min_ts']
device_stats['con_max_ts'] = device_conn_stats_tsec['con_max_ts']
device_stats['con_mode_ts'] = device_conn_stats_tsec['con_mode_ts']
device_stats['con_mean_ts'] = device_conn_stats_tsec['con_mean_ts']

# disconnect stats
device_stats['dcon_min_ts'] = device_disconn_stats_tsec['dcon_min_ts']
device_stats['dcon_max_ts'] = device_disconn_stats_tsec['dcon_max_ts']
device_stats['dcon_mode_ts'] = device_disconn_stats_tsec['dcon_mode_ts']
device_stats['dcon_mean_ts'] = device_disconn_stats_tsec['dcon_mean_ts']

# files per day stats
device_stats['file_mode'] = files_per_day_stats['mode']
device_stats['file_max'] = files_per_day_stats['max']

device_stats.head()

# save the data
device_stats.to_csv('./Data_Subset/Input_features/device_stats.csv', index = False)

# input array
device_stats.columns[1:]

device_params = device_stats.as_matrix(columns = device_stats.columns[1:])
device_params

# fit the model
device_params = device_params #.reshape(-1,1)
forest = IsolationForest()

forest.fit(device_params)

# anomaly score

dev_file_ascore = forest.decision_function(device_params)
dev_file_ascore

dev_file_ascore.shape

device_stats.shape

# Save the result
device_file_result = pd.DataFrame()

device_file_result['user'] = device_stats['user']
device_file_result['ascore'] = dev_file_ascore

device_file_result.head()

# save the result
device_file_result.to_csv('./Data_Subset/IFResult/device_file_result.csv', index = False)

psychometric = pd.read_csv('./Data_Subset/fu2_psychometric.csv')

psychometric.head()

psychometric.shape

# fit the model

# input array
psychometric_params = psychometric.as_matrix(columns = psychometric.columns[2:])
psychometric_params

#device_params = device_params
forest = IsolationForest()

forest.fit(psychometric_params)

# anomaly score
psych_ascore = forest.decision_function(psychometric_params)
psych_ascore

psych_ascore.shape

# save the result
psychometric_result = pd.DataFrame()

psychometric_result['user'] = psychometric['user_id']
psychometric_result['ascore'] = psych_ascore

psychometric_result.head()

psychometric_result.to_csv('./Data_Subset/IFResult/psychometric_result.csv', index = False)

dfmerge1 = pd.merge(ulog_on_off_stats, user_pc, on = 'user')
dfmerge1.head()

dfmerge2 = pd.merge(dfmerge1, psychometric, left_on = 'user', right_on = 'user_id')
dfmerge2.head()

dfmerge2.shape

# drop the 'employee_name' and 'user_id' columns
dfmerge2.drop(['employee_name', 'user_id'], axis=1, inplace=True)  # axis: 1 for col, 0 for row

dfmerge2.head()

All_params = dfmerge2

# save
All_params.to_csv('./Data_Subset/Input_features/All_params.csv', index = False)

#input array

All_params_input = All_params.as_matrix(columns = All_params.columns[1:])
All_params_input

forest = IsolationForest()

forest.fit(All_params_input)

# Anomaly score
All_params_ascore = forest.decision_function(All_params_input)
All_params_ascore

# save the result

All_params_result = pd.DataFrame()

All_params_result['user'] = All_params['user']
All_params_result['ascore'] = All_params_ascore

All_params_result.head()

All_params_result.to_csv('./Data_Subset/IFResult/All_params_result.csv', index = False)



