import numpy as np
import pandas as pd
import matplotlib as plt
get_ipython().magic('matplotlib inline')

diag = pd.read_csv("./raw_mimic_data/DIAGNOSES_ICD.csv", dtype = str)

diag.head(5)

print diag.shape[0]
print diag.ICD9_CODE.nunique()
print diag.SUBJECT_ID.nunique()

#set wether to group icd 9 codes (first 3 string)
group_icd_codes = True

if group_icd_codes:
    diag["ICD9_CODE"] = diag["ICD9_CODE"].apply(lambda x: str(x)[:3])

#plenty of people diagnosed HF
print diag[diag["ICD9_CODE"] == "D428"].SUBJECT_ID.nunique()

#here I use first 3 digits to group diagonisis codes to avoid sparsity
diag_clean = diag.ix[:,["SUBJECT_ID", "ICD9_CODE"]]
diag_count = diag_clean.groupby("ICD9_CODE").count()

#124 codes have 5 instances or less. For now I will include them in the model but will check in the future
print diag_count.shape
print diag_count[diag_count.SUBJECT_ID < 5].shape

#now need to find a way to get the date and time 
#import boto
#join admission time as the time of the diagonsis
admission = pd.read_csv("./raw_mimic_data/ADMISSIONS.csv", dtype = str)
admission = admission.ix[:,["SUBJECT_ID", "HADM_ID", "ADMITTIME"]]

diag_w_time = diag.merge(admission, how = "inner", on=["SUBJECT_ID", "HADM_ID"])
diag_w_time.rename(columns = {"ADMITTIME": "TIME", "ICD9_CODE":"EVENTS"}, inplace = True )

diag_w_time.shape

diag_clean = diag_w_time.ix[:,["SUBJECT_ID", "EVENTS", "TIME"]]

#filter out all empty or null diag codes
diag_clean["EVENTS"] = diag_clean.EVENTS.str.strip()
diag_clean = diag_clean[~diag_clean.EVENTS.isnull()]
diag_clean = diag_clean[diag_clean.EVENTS != ""]
diag_clean.shape

#make sure no empty/null events
print diag_clean[diag_clean.EVENTS.apply(lambda x: len(x.strip())) == 0].shape[0]
print diag_clean[diag_clean.EVENTS.isnull()].shape[0]

#diagnosis data with time
diag_clean.head()

#now I will look at the prescription
pres = pd.read_csv("./raw_mimic_data/PRESCRIPTIONS.csv", dtype = str)

#again extremely skewed distribution, might need to group rare codes 
pres_clean = pres.ix[:, ["SUBJECT_ID", "STARTDATE", "DRUG"]]
drug_ct = pres_clean.groupby("DRUG").count()["SUBJECT_ID"]
drug_ct.describe()

pres_clean.shape[0]

pres_clean.rename(columns = {"STARTDATE":"TIME", "DRUG":"EVENTS"},inplace = True)

#get rid of all null/empty observations
pres_clean["EVENTS"] = pres_clean.EVENTS.str.strip()
pres_clean["EVENTS"] = pres_clean.EVENTS.str.lower()

pres_clean = pres_clean[~pres_clean.EVENTS.isnull()]
pres_clean = pres_clean[pres_clean.EVENTS != ""]

#make sure no empty/null events
print pres_clean[pres_clean.EVENTS.apply(lambda x: len(x.strip())) == 0].shape[0]
print pres_clean[pres_clean.EVENTS.isnull()].shape[0]
print pres_clean.shape[0]

pres_clean.head()

proc = pd.read_csv("./raw_mimic_data/PROCEDURES_ICD.csv", dtype = str)

proc_w_time = proc.merge(admission, how = "inner", on=["SUBJECT_ID", "HADM_ID"])
proc_w_time.rename(columns = {"ADMITTIME": "TIME", "ICD9_CODE":"EVENTS"}, inplace = True )

proc_w_time.shape[0]

proc_clean = proc_w_time.ix[:,["SUBJECT_ID", "TIME","EVENTS"]]
proc_clean.head()

# no empty/null value in procesdure data set
print proc_clean[proc_clean.EVENTS.isnull()].shape[0]
print proc_clean[proc_clean.EVENTS.apply(lambda x: len(x.strip())) == 0].shape[0]

if group_icd_codes:
    proc_clean["EVENTS"] = proc_clean["EVENTS"].apply(lambda x: str(x)[:3])

proc_clean.TIME.min()

proc_events = pd.read_csv("./raw_mimic_data/PROCEDUREEVENTS_MV.csv", dtype = str)

proc_events.head()

proc_events[proc_events.ORDERCATEGORYDESCRIPTION == "Electrolytes"].ORDERCATEGORYNAME.value_counts()

proc_events.ORDERCATEGORYDESCRIPTION.value_counts()

proc_clean_MV = proc_events.ix[proc_events.ORDERCATEGORYDESCRIPTION == "Electrolytes", ["SUBJECT_ID", "STARTTIME", "ITEMID"]]
item_id = pd.read_csv("./raw_mimic_data/D_ITEMS.csv", dtype = str)
item_id = item_id.ix[:,["ITEMID", "LABEL"]]
proc_clean_MV = proc_clean_MV.merge(item_id, on = "ITEMID")

proc_clean_MV.rename(columns = {"STARTTIME" : "TIME","LABEL": "EVENTS"}, inplace = True)
proc_clean_MV = proc_clean_MV.drop(["ITEMID"], axis = 1)

print proc_clean_MV[proc_clean.EVENTS.isnull()].shape[0]
print proc_clean_MV[proc_clean.EVENTS.apply(lambda x: len(x.strip()) == 0)].shape[0]

proc_clean_MV["EVENTS"] = proc_clean_MV.EVENTS.str.strip()
proc_clean_MV["EVENTS"] = proc_clean_MV.EVENTS.str.lower()

proc_clean_MV.TIME.min()

use_mv = False

if use_mv:
    all_events_data = pd.concat([diag_clean,proc_clean_MV, pres_clean], axis = 0)
else:
    all_events_data = pd.concat([diag_clean,proc_clean, pres_clean], axis = 0)

all_events_data.shape[0]

from sklearn.utils import shuffle
#randomly shuffle the data so it breaks the sequence order with diag and procedure with same subject id and time
all_events_data = shuffle(all_events_data).reset_index(drop=True)

#turn medication, diag, procedure events into integer ids (by their alphabetical order)
all_events = all_events_data.EVENTS.unique()
all_events.sort()
index = np.arange(1,len(all_events)+1)
events_lookup = pd.DataFrame({"EVENTS":all_events, "EVE_INDEX":index})

if use_mv:
    events_lookup.to_csv("./cleaned_data/events_id_mv.csv")
else:
    events_lookup.to_csv("./cleaned_data/events_id.csv")

#merge all_events_data with event_index
all_events_data = all_events_data.merge(events_lookup, on = "EVENTS")

all_events_data = all_events_data.sort_values(by = ["SUBJECT_ID", "TIME"]).reset_index(drop=True)
all_events_data.head()

if use_mv:
    all_events_data.to_csv("./cleaned_data/all_events_data_mv.csv")
else:
    all_events_data.to_csv("./cleaned_data/all_events_data.csv")

eve_value_count = all_events_data.EVE_INDEX.value_counts()

eve_value_count[eve_value_count < 2].shape

eve_value_count.describe()

