import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

with_time = False
if with_time:
    all_events = pd.read_csv('./cleaned_data/all_events_data_w_time.csv')
else:
    all_events = pd.read_csv('./cleaned_data/all_events_data.csv')

all_events.head()

np.unique(all_events.SUBJECT_ID).shape

# event counts of all ids

all_count = all_events.groupby('SUBJECT_ID')['EVE_INDEX'].agg({'count':lambda x: len(x)})

all_count.reset_index(inplace=True)

all_count.head()

all_count.describe()

# Control / Case group
case_id = pd.unique(all_events.ix[all_events['EVENTS'].str[:10] == '###diag428','SUBJECT_ID'])
control_id =np.setdiff1d(np.unique(all_events['SUBJECT_ID'].values),case_id)

case_id.shape

control_id.shape

control = all_events.ix[np.in1d(all_events['SUBJECT_ID'],control_id),:]
case = all_events.ix[np.in1d(all_events['SUBJECT_ID'],case_id),:]

control.head()

np.unique(control.SUBJECT_ID).shape

np.unique(case.SUBJECT_ID).shape

observation_window = 2000

control_index_date = control.groupby(['SUBJECT_ID'],                     as_index=False)['TIME'].agg({'INDEX_DATE': lambda x: pd.to_datetime(x).max()})

control_filter = pd.merge(control,control_index_date,how='left',on = ['SUBJECT_ID'])

#observation window is set to be 2000 days
choice1 = pd.to_datetime(control_filter['TIME'])>= pd.to_datetime(control_filter['INDEX_DATE']) - pd.DateOffset(observation_window)
choice2 = pd.to_datetime(control_filter['TIME']) <= pd.to_datetime(control_filter['INDEX_DATE'])
control_filter = control_filter[choice1 & choice2]
control_filter.head()    

# control count
count_control = control_filter.groupby('SUBJECT_ID').apply(lambda x: x.EVE_INDEX.size)
count_control.hist(bins=100)
print(np.histogram(count_control))
print(np.mean(count_control)); print(np.median(count_control)); print(np.std(count_control))

# filter out control IDs with count >500 & count<30 ??
control_filter =pd.merge(control_filter, all_count,on='SUBJECT_ID',how='left')
control_filter_out = control_filter.ix[((control_filter['count']>30) & (control_filter['count']<500)),:]

np.unique(control_filter_out.SUBJECT_ID).shape

control_filter_out.head()

prediction_window = 90

case_index_date = case.ix[case['EVENTS'].str[:10]=='###diag428',:].groupby(['SUBJECT_ID'],                         as_index=False)['TIME'].agg({'INDEX_DATE':                         lambda x: pd.to_datetime(x).min()- pd.Timedelta(days = prediction_window)})

case_index_date.head()

case_filter = pd.merge(case,case_index_date,how='left',on = ['SUBJECT_ID'])

case_filter.head()

choice1 = pd.to_datetime(case_filter['TIME'])>= pd.to_datetime(case_filter['INDEX_DATE']) - pd.DateOffset(2000)
choice2 = pd.to_datetime(case_filter['TIME']) <= pd.to_datetime(case_filter['INDEX_DATE'])
case_filter = case_filter[choice1 & choice2]
case_filter.head()    

# case count
count_case = case_filter.groupby('SUBJECT_ID').apply(lambda x: x.EVE_INDEX.size)
count_case.hist(bins=100)
print(np.histogram(count_case))
print(np.mean(count_case)); print(np.median(count_case)); print(np.std(count_case))

count_case.describe()

# filter out <10 & > 1000???

case_filter =pd.merge(case_filter, all_count,on='SUBJECT_ID',how='left')
case_filter_out = case_filter.ix[((case_filter['count']>30) & (case_filter['count']<500)),:]

case_filter_out.head()

np.unique(case_filter_out.SUBJECT_ID).size

case_filter_out.shape

# get sample from control
np.random.seed(6250)
sample_ids = np.random.choice(np.unique(control_filter_out.SUBJECT_ID),2*np.unique(case_filter_out.SUBJECT_ID).size,replace=False)

control_out = control_filter_out.ix[np.in1d(control_filter_out['SUBJECT_ID'],sample_ids),:]

np.unique(control_out.SUBJECT_ID).size

control_out.reset_index(inplace=True)

if with_time:
    control_out.ix[:,['SUBJECT_ID','TIME','EVE_INDEX']].to_csv('./cleaned_data/control_w_time.csv')
    case_filter_out.ix[:,['SUBJECT_ID','TIME','EVE_INDEX']].to_csv('./cleaned_data/case_w_time.csv')
else:
    control_out.ix[:,['SUBJECT_ID','TIME','EVE_INDEX']].to_csv('./cleaned_data/control.csv')
    case_filter_out.ix[:,['SUBJECT_ID','TIME','EVE_INDEX']].to_csv('./cleaned_data/case.csv')

demographic = pd.read_csv("./cleaned_data/demographic.csv",index_col = 0)

def get_demographic(data):
    demo = data.merge(demographic, how = "left", on = "SUBJECT_ID")
    demo = demo.ix[:, ["SUBJECT_ID", "INDEX_DATE", "GENDER", "DOB", "ETHNICITY"]].drop_duplicates()
    #calculate the age for the patients
    age = demo.INDEX_DATE - pd.to_datetime(demo.DOB)
    demo["AGE"] = age.apply(lambda x: x.days/365.25 +300 if x.days <0 else x.days/365.25)
    demo = demo.drop(["DOB", "INDEX_DATE"], axis = 1)
    demo_full = demo.merge(pd.get_dummies(demo.ETHNICITY), left_index= True, right_index= True)
    #gender 1 is M 0 is F
    demo_full.GENDER = demo_full.GENDER.apply(lambda x: 1 if x == "M" else 0)
    
    return demo_full.drop(["ETHNICITY"], axis = 1)

if with_time:
    control_demo = get_demographic(control_out)
    control_demo.to_csv('./cleaned_data/control_demo_wt.csv')
    case_demo = get_demographic(case_filter_out)
    case_demo.to_csv('./cleaned_data/case_demo_wt.csv')
else:
    control_demo = get_demographic(control_out)
    control_demo.to_csv('./cleaned_data/control_demo.csv')
    case_demo = get_demographic(case_filter_out)
    case_demo.to_csv('./cleaned_data/case_demo.csv')







