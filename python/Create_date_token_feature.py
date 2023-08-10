import numpy as np
import pandas as pd

use_mv = True

if use_mv:
    all_events_data = pd.read_csv("./cleaned_data/all_events_data_mv.csv",index_col = 0,dtype = str)
else:
    all_events_data = pd.read_csv("./cleaned_data/all_events_data.csv", index_col = 0,dtype = str)

all_events_data["EVE_INDEX"] = all_events_data["EVE_INDEX"].astype("int")

all_events_data["TIME"] =pd.to_datetime(all_events_data["TIME"],infer_datetime_format = True,format="%Y-%m-%d %H:%M:%S")
#delete all invalid time
all_events_data =all_events_data[~all_events_data.TIME.isnull()].reset_index(drop = True)

#merge with timestamp of next event
time_next = pd.DataFrame({"TIME_next":all_events_data.TIME[1:]}).reset_index(drop = True)
time_merged = all_events_data.merge(time_next, how = "left", left_index=True, right_index=True)

#calculate day-gaps after each event for each 
time_token =pd.DataFrame({"time_gap":time_merged.TIME_next - time_merged.TIME, "flag":np.ones(len(time_merged))})

time_token = all_events_data.merge(time_token, how = "left", left_index=True, right_index=True)

#get rid of the last time-gap of each patient (its meaningless)
time_token_clean = time_token.groupby("SUBJECT_ID").apply(lambda x: x[:-1])

#remove the observation with inconsistant time. one happend over 50 years before the next event
a = time_token_clean[time_token_clean.time_gap > pd.Timedelta(days=15000)]
index_to_drop = [i for (a,i) in a.index]
#remove from the orginal data,
all_events_data.drop(index_to_drop, inplace = True)
#remove from time token data
time_token_clean = time_token_clean[time_token_clean.time_gap < pd.Timedelta(days=15000)]
#reset_index
time_token_clean.reset_index(drop = True, inplace = True)

time_token_clean.time_gap = time_token_clean.time_gap.apply(lambda x:x.days)

# to create tokens only take time gap >0
time_token_clean = time_token_clean[time_token_clean.time_gap > 0]

#create bins 0-2 days 3-5 days 6-12 days 13-30 days 30-90 days 90-365 days 365+ days
max_gap_days = np.max(time_token_clean.time_gap)
max_index = all_events_data["EVE_INDEX"].max()

time_token_clean["Bin_indx"] = pd.cut(time_token_clean.time_gap, [0,2,7,15,90,365,max_gap_days], labels=[max_index+1,max_index+2,max_index+3,max_index+4,max_index+5,max_index+6])
time_token_clean["Bin"] = pd.cut(time_token_clean.time_gap, [0,2,7,15,90,365,max_gap_days],                                  labels=["timetoken0-2day","timetoken3-7day","timetoken8-15day","timetoken16-90day","timetoken91-365day","timetoken366+day"])

time_token_clean = time_token_clean.drop(["EVE_INDEX", "EVENTS"], axis=1)                            .rename(columns = {"Bin_indx":"EVE_INDEX","Bin":"EVENTS" })

#create time tokens as "events" and "event" index same format as other events
time_token_clean_final = time_token_clean.reindex(columns = [u'EVENTS', u'SUBJECT_ID', u'TIME', u'EVE_INDEX','flag'])
time_token_clean_final['EVENTS'] = time_token_clean_final['EVENTS'].astype('object')
time_token_clean_final['EVE_INDEX'] = time_token_clean_final['EVE_INDEX'].astype('int')

#concatenate with event data
all_events_data["flag"] = np.zeros(len(all_events_data))
all_events_w_time = pd.concat([time_token_clean_final,all_events_data], axis = 0)                            .sort_values(by =['SUBJECT_ID','TIME','flag'])

all_events_w_time.drop(['flag'],axis = 1, inplace = True)

if use_mv:
    all_events_w_time.to_csv("./cleaned_data/all_events_data_w_time_mv.csv")
else:
    all_events_w_time.to_csv("./cleaned_data/all_events_data_w_time.csv")

event_id = all_events_w_time.ix[:,["EVE_INDEX","EVENTS"]].sort_values(by = "EVE_INDEX").drop_duplicates()
if use_mv:
    event_id.to_csv("./cleaned_data/events_id_w_time_mv.csv")
else:
    event_id.to_csv("./cleaned_data/events_id_w_time.csv")



