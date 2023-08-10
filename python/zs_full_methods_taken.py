import pandas as pd
import numpy as np

train = pd.read_csv("train.csv")
testc = pd.read_csv("test.csv")
sample = pd.read_csv("sample_submission.csv")

train.head()

test.head()

sample.head()

print train.shape,test.shape

np.setdiff1d(train.PID,test.PID)

len(train.Event.unique())

len(train.PID.unique())

(test.PID.value_counts()>1).describe()

(train.PID.value_counts()>50).describe()

train.loc[train['PID'] == 1029677].Event

event_counts = train.Event.value_counts()
len(event_counts[event_counts>40])

event_counts_1011411 = train.loc[train['PID'] == 1011411].Event.value_counts()
event_counts_1011411

### 1. Fixing jan 1, 2014 to check age of a record

from datetime import datetime

date_threshold = datetime.strptime("January, 2014","%B, %Y")

train["Date"] = pd.to_datetime(train["Date"],format="%Y%m")

train["record_age"] = train.Date.apply(lambda x: (date_threshold - x).days)

train.head()

#Find Records of last x months and sort them by date
train_last_x_months = train.loc[train['record_age']<=1010]
train_last_x_months.sort_values(by="Date",ascending=False,inplace=True)

train_last_x_crosstab = pd.crosstab(index=train_last_x_months['PID'], columns=train_last_x_months['Event'])

len(train_last_x_months.PID.unique())

np.setdiff1d(test["PID"],train_last_x_months["PID"])

# train_last_six_crosstab = pd.crosstab(index=train_last_six_months['PID'], columns=train_last_six_months['Event'])
submit3 = train_last_x_crosstab.loc[:,train_last_x_crosstab.columns != 'PID'].apply(lambda x: pd.Series(x.sort_values(ascending=False).iloc[:10].index, index=['Event'+str(x) for x in range(1,11)]),axis=1).reset_index()

submit3.reset_index(drop=True)

submit3.to_csv("third_sub.csv",index=False)

train_2013 = train.loc[train['record_age']<=365]

PID_left = np.setdiff1d(train.PID,train_2013.PID)
train_2013_left_ids = train.loc[train['PID'].isin(PID_left)]

train_latest = pd.concat([train_2013,train_2013_left_ids],axis=0)

len(train_latest.PID.unique())

train_latest_crosstab = pd.crosstab(index=train_latest['PID'],columns=train_latest['Event'])

submit4 = train_latest_crosstab.loc[:,train_latest_crosstab.columns != 'PID'].apply(lambda x: pd.Series(x.sort_values(ascending=False).iloc[:10].index, index=['Event'+str(x) for x in range(1,11)]),axis=1).reset_index()

submit4.to_csv('fourth_submit.csv',index=False)

train_six_months = train.loc[train['record_age']<=300]

PID_left_six = np.setdiff1d(train.PID,train_six_months.PID)
train_six_left_ids = train.loc[(train['PID'].isin(PID_left_six)) & (train['record_age']<=1010)]

train_six = pd.concat([train_six_months,train_six_left_ids],axis=0)

len(train_six.PID.unique())

train_six_crosstab = pd.crosstab(index=train_six['PID'],columns=train_six['Event'])

submit5 = train_six_crosstab.loc[:,train_six_crosstab.columns != 'PID'].apply(lambda x: pd.Series(x.sort_values(ascending=False).iloc[:10].index, index=['Event'+str(x) for x in range(1,11)]),axis=1).reset_index()

submit5.to_csv('twelve_submit.csv',index=False)

train_six_months_order = train.loc[train['record_age']<=180]
PID_left_six_order = np.setdiff1d(train.PID,train_six_months_order.PID)
train_six_order_left_ids = train.loc[train['PID'].isin(PID_left_six)]
train_six_order = pd.concat([train_six_months_order,train_six_order_left_ids],axis=0)

len(train_six_order.PID.unique())

train_six_order_crosstab = pd.crosstab(index=train_six_order['PID'],columns=train_six_order['Event'])

submit6 = train_six_order_crosstab.loc[:,train_six_order_crosstab.columns != 'PID'].apply(lambda x: pd.Series(x.iloc[-10:].index, index=['Event'+str(x) for x in range(1,11)]),axis=1).reset_index()

submit6.to_csv('six_submit.csv',index=False)

train_two_months = train.loc[train['record_age']<=60]

len(train_two_months.PID.unique())

pid_left_two_months = np.setdiff1d(train.PID,train_two_months.PID)

train_two_months_left = train.loc[train['PID'].isin(pid_left_two_months)]

train_two = pd.concat([train_two_months,train_two_months_left],axis=0)

len(train_two.PID.unique())

train_two_crosstab = pd.crosstab(index=train_two['PID'],columns=train_two['Event'])

submit7 = train_two_crosstab.loc[:,train_two_crosstab.columns != 'PID'].apply(lambda x: pd.Series(x.sort_values(ascending=False).iloc[:10].index, index=['Event'+str(x) for x in range(1,11)]),axis=1).reset_index()

submit7.to_csv('seven_submit.csv',index=False)

train_6 = train.loc[train['record_age']<=180]

missing_ids_6 = np.setdiff1d(train.PID,train_6.PID)
print len(missing_ids_6)
missing_values = train[train.PID.isin(missing_ids_6)]

train_23 = train.loc[(train['record_age']<=700)&train['PID'].isin(missing_ids_6)]

missing_ids_23 = np.setdiff1d(missing_ids_6,train_23.PID)

len(missing_ids_23)

train_27 = train.loc[(train['record_age']<=1010)&train['PID'].isin(missing_ids_23)]

train_multi_rule = pd.concat([train_6,train_23,train_27],axis=0)

len(train_multi_rule.PID.unique())

train_multi_crosstab = pd.crosstab(index=train_multi_rule['PID'], columns=train_multi_rule.Event)

submit8 = train_multi_crosstab.loc[:,train_multi_crosstab.columns != 'PID'].apply(lambda x: pd.Series(x.sort_values(ascending=False).iloc[:10].index, index=['Event'+str(x) for x in range(1,11)]),axis=1).reset_index()

submit8.to_csv("eight_sub.csv",index=False)

train_10 = train.loc[train['record_age']<=300]

missing_ids_10 = np.setdiff1d(train.PID,train_x.PID)

train_missing_10 = train.loc[train['PID'].isin(missing_ids_10)]

train_10_final = pd.concat([train_10,train_missing_10],axis=0)

len(train_10_final.PID.unique())

train_10_crosstab = pd.crosstab(index=train_10_final.PID, columns=train_10_final.Event)

submit9 = train_10_crosstab.loc[:,train_10_crosstab.columns != 'PID'].apply(lambda x: pd.Series(x.sort_values(ascending=False).iloc[:10].index, index=['Event'+str(x) for x in range(1,11)]),axis=1).reset_index()

submit9.to_csv('nine_sub.csv',index=False)

record_count = train.PID.value_counts()

train.PID.value_counts().describe()

train.PID.value_counts().mode()

train['record_count'] = train.PID.apply(lambda x: record_count[x])

len(train.loc[train['record_count']<50].PID.unique())

train_by_pid = train.groupby('PID')
train_last_50 = train_by_pid.tail(500)

len(train_last_50.PID.unique())

train_group_crosstab = pd.crosstab(index=train_last_50['PID'],columns=train_last_50['Event'])

submit10 = train_group_crosstab.loc[:,train_group_crosstab.columns != 'PID'].apply(lambda x: pd.Series(x.sort_values(ascending=False).iloc[:10].index, index=['Event'+str(x) for x in range(1,11)]),axis=1).reset_index()

submit10.to_csv('submit10.csv',index=False)



