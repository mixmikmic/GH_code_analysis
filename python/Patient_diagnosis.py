import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

csv_directory = "/home/ubuntu/zehai/data/synpuf/"
json_directory = "/home/ubuntu/zehai/data/AchillesExport/"

## Load the condition_occation file from the directory
condition_occur = pd.read_csv(csv_directory +"condition_occurrence.csv",dtype={'condition_concept_id':'int64',
 'condition_end_date': 'O',
 'condition_occurrence_id':'int64',
 'condition_source_concept_id':'int64',
 'condition_source_value':'O',
 'condition_start_date':'O',
 'condition_type_concept_id':'int64',
 'person_id':'int64',
 'provider_id':'float64',
 'stop_reason': 'float64',
 'visit_occurrence_id': 'int64'}, nrows = 43367979 )

#read the condition map from Json file
condition_map = pd.read_json(json_directory+"condition_treemap.json")

# read the observation file
observation = pd.read_csv(csv_directory+"observation.csv", dtype = {'observation_concept_id': 'int64',
 'observation_date': 'O',
 'observation_id': 'int64',
 'observation_source_concept_id': 'int64',
 'observation_source_value': 'O',
 'observation_time': 'float64',
 'observation_type_concept_id': 'int64',
 'person_id': 'int64',
 'provider_id': 'float64',
 'qualifier_concept_id': 'float64',
 'qualifier_source_value': 'float64',
 'unit_concept_id': 'float64',
 'unit_source_value': 'float64',
 'value_as_concept_id': 'int64',
 'value_as_number': 'float64',
 'value_as_string': 'float64',
 'visit_occurrence_id': 'int64'})

# Load brain antonmy keys from txt fiel
with open('brain_antonomy_keys.txt') as f:
     read_data = f.read()
f.closed
antonomy_regions = [x.strip() for x in read_data.split(',')]

# read the observation map
ob_map = pd.read_json(json_directory+"observation_treemap.json")

# conditionera map from Json dir
conditionera_map = pd.read_json(json_directory+"conditionera_treemap.json")

# read the Head MRI related patient from the oberservation dataset
#      CONCEPT_ID                                       CONCEPT_PATH  \   
# 121     2106371  Radiography of head||Regional contrast radiolo...   
# 218     2106382  Radiography of head||Regional contrast radiolo...     

#      NUM_PERSONS  PERCENT_PERSONS  RECORDS_PER_PERSON   
# 121         1137         0.009772            1.117854  
# 218          519         0.004461            1.015414   

brain_identify_patient = observation[(observation['observation_concept_id']==2106371)
                                     |(observation['observation_concept_id']==2106382)]
patient_id = brain_identify_patient['person_id'].drop_duplicates()

# select all the patients with head mri scan history
mri_condition_occur = condition_occur[condition_occur['person_id'].isin(list(patient_id))]

# de_duplicate condition_concept, select all the symptoms related to the MRI patients
mri_conditions = mri_condition_occur.loc[:,['person_id','condition_concept_id']]
head_condition = mri_conditions.condition_concept_id.drop_duplicates()
mri_related_symptom = condition_map['CONCEPT_PATH'][condition_map.CONCEPT_ID.isin(list(head_condition))]

# load brain antonmy keys from txt fiel
with open('brain_antonomy_keys.txt') as f:
     read_data = f.read()
f.closed
antonomy_regions = [x.strip() for x in read_data.split(',')]

# converted everything to lower case for matching
antonomy_regions = [x.lower() for x in antonomy_regions]
conditionera_map['CONCEPT_PATH'] = conditionera_map['CONCEPT_PATH'].str.lower()

# match the conditionera SNOMED discriptions with the brain section regions
brain_related_map = conditionera_map.loc[:,['CONCEPT_ID','CONCEPT_PATH']][sum([conditionera_map['CONCEPT_PATH'].str.contains(x) for x in antonomy_regions])>0]

# select all mri patient with final brain issue diagnosis
brain_related_list = list(brain_related_map['CONCEPT_ID'])
brain_problem_diagnosis = mri_conditions.loc[:,['person_id','condition_concept_id']][mri_conditions['condition_concept_id'].isin(brain_related_list)]
brain_problem_patients = brain_problem_diagnosis.drop_duplicates()
brain_problem_diagnosis['diagnosis'] = 0
brain_related_list = list(brain_related_map['CONCEPT_ID'])
brain_diagnosis = list(brain_related_map['CONCEPT_PATH'].str.split('|', expand=True)[8])
for i in range(len(brain_related_list)):
    brain_problem_diagnosis['diagnosis'][brain_problem_diagnosis['condition_concept_id']== brain_related_list[i]] = brain_diagnosis[i]

# selected all symptoms related to brain issue
mri_symptoms = mri_conditions[mri_conditions.condition_concept_id.isin(brain_related_list)==False].drop_duplicates()
symptoms_of_braindamage = mri_symptoms.condition_concept_id.drop_duplicates()
symptoms_of_braindamage.head()

# selected specific symptom with brain issue population
symptomlist = condition_map.loc[:,['CONCEPT_ID','CONCEPT_PATH']][condition_map['CONCEPT_ID'].isin(list(symptoms_of_braindamage))]
symptomlist = symptomlist.reset_index()
symptomlist.head()

# read in all the neurosynth dictionary
with open('/home/ubuntu/zehai/data/NS_TERMS.txt') as f:
     read_data = f.read()
f.closed
neuro_keylist = [x.strip() for x in read_data.split('\n')]

brain_problem_diagnosis_deduplic.head()

patient_symptoms.head()

patient_symptoms = mri_symptoms.groupby('person_id')['condition_concept_id'].apply(list).reset_index()
brain_problem_diagnosis_deduplic = brain_problem_diagnosis.drop_duplicates()
final_parse = pd.merge(brain_problem_diagnosis_deduplic, patient_symptoms, on='person_id', how='inner')
final_parse.head()
final_parse.rename(columns={'condition_concept_id_x': 'diagnosis_id', 'condition_concept_id_y': 'symptom_id_list'}, inplace=True)
final_parse.head()

def keys_append(concept_path,neuro_keylist):
    # append neurosynth keys to symptom map
    concept_path = concept_path.lower()
    keylist = []
    for x in neuro_keylist:
        # String length greater than 3 is totally arbitrary
        # But it makes the resulting lists look way better, so
        # For the sake of expediency, we're going with it
        # TODO: get a better heuristic here
        if concept_path.find(' ' + x) != -1 and len(x) > 3:
            keylist.append(x)
    #keylist = ', '.join(keylist)
    return keylist

symp_list = []
for i in list(symptomlist.CONCEPT_PATH):
    #list of keys concatenated
    symp_list.append(keys_append(i, neuro_keylist))

# Add symptom key_list to patient table
sl = pd.Series( (v for v in symp_list) )
# symptomlist = symptomlist.drop(columns = 'index')
symptomlist['keys_list'] = sl
symptomlist.tail()

len(keylist)
list(symptomlist['keys_list'][symptomlist['CONCEPT_ID']==434509])[0]

def key_match(keylist,symptomlist):
    # match all the keylist with the symptom to neurosynth dict
    # translate symptoms to key
    keys_per_symp = []
    for i in keylist:
        thinging = list(symptomlist['keys_list'][symptomlist['CONCEPT_ID']==i])
        if thinging==[]:
            keys_per_symp.append(thinging)
        else:
            keys_per_symp.append(thinging[0])
    return keys_per_symp

## test on the first patient
# keylist = final_parse.symptom_id_list[0]
# key_match(keylist,symptomlist)

# type(symptomlist['keys_list'][symptomlist['CONCEPT_ID']==440977])

keys_per_person =[]
for x in list(final_parse.symptom_id_list):
        keys_per_person.append(key_match(x,symptomlist))

ssl = pd.Series( (v for v in keys_per_person) )
final_parse['neurosynth_keys'] = ssl
final_parse.head()

symp_map = symptomlist.drop(columns='index')
symp_map.head()

# expand all symptoms
symptom_list = final_parse.loc[:,['person_id','symptom_id_list']]
symptom_list.head()

neuro_list = final_parse.loc[:,['person_id','neurosynth_keys']]
neuro_list.head()

s = symptom_list.apply(lambda x: pd.Series(x['symptom_id_list']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'symptom_id'
s = s.apply(int)

n = neuro_list.apply(lambda x: pd.Series(x['neurosynth_keys']),axis=1).stack().reset_index(level=1, drop=True)
n.name = 'neurosynth_key'
n.head()

combine = pd.concat([s, n], axis=1)
symptom_list.head()

second_parse = final_parse.loc[:,['person_id','diagnosis_id','diagnosis']]

second_parse = second_parse.join(combine)
second_parse.head()

# delete all symptoms with no neurosynth key match
# third_parse = second_parse[second_parse['neurosynth_key'].apply(len())]
second_parse = second_parse.reset_index(drop=True)
def isempty(x):
    if x==[]:
        return False
    else:
        return True

third_parse = second_parse[second_parse['neurosynth_key'].apply(isempty)]
third_parse.count()

to_save = third_parse.merge(symp_map.loc[:,['CONCEPT_ID','CONCEPT_PATH']].rename(columns={'CONCEPT_ID':'symptom_id'}), how='left', on='symptom_id')

to_save['symptom_name'] = to_save.CONCEPT_PATH.str.split('|').str[-1]

get_ipython().system(' mkdir ./clint/docs/example_data')

to_save.to_pickle('./clint/docs/example_data/symptom_table_1.pkz')

# Get a list of all symptom feature name
# symp_map['CONCEPT_PATH'][symp_map['CONCEPT_ID']==193136]
symp_name = []
for i in list(third_parse['symptom_id']):
    symp_name.append(list(symp_map['CONCEPT_PATH'][symp_map['CONCEPT_ID']==i]))

symp_name_list = pd.Series( (v for v in symp_name) )
symp_name_list.head()

neurosynth_parse = final_parse.drop(columns = ['diagnosis_id'])
neurosynth_parse.tail()

# Clean all the empty symptom keys
import pickle as pk

file_Name = "patient_symptoms_3"
# open the file for writing
fileObject = open(file_Name,'wb') 

# this writes the object a to the
# file named 'testfile'
# pk.dump(neurosynth_parse,fileObject)   
pk.dump(third_parse,fileObject)   

# here we close the fileObject
fileObject.close()

import pickle

def get_label_map():
    levels = ['region', 'hemisphere_region', 'lobe', 'gyrus', 'tissue', 'sub-label']
    label_map = {}
    for level in levels:
        with open ('/home/ubuntu/zehai/clint/data/%s_label_map.pkz'%level, 'rb') as h:
            label_map.update(pickle.load( h))
    return label_map
brain_label_map = get_label_map()

# convert the map list to lower case
map_list = [x for x in list(brain_label_map.keys())]
a = map_list[0]
print (a)
# brain_label_map[a]

map_dim = np.shape(brain_label_map[a])

third_parse.head()

test_set = third_parse[:10]
test_set.count()

map_list

not not check_share_lexi('Brodmann area 23','Brodmannnnn area 25')

# Map the brain diagnosis with brain map
# for i in map_list: # loop over the entier map_list

def check_share_lexi(a,b):
    # check if two string lists overlap in a word sense
    a = a.lower()
    b = b.lower()
    return list(set(a.split()) & set(b.split()))

pic = []
for patient_diag in list(third_parse.diagnosis):
    person_map = np.zeros(map_dim)
    for key in map_list:
        if not not check_share_lexi(key,patient_diag):
            person_map = np.logical_or(person_map,brain_label_map[key])
    pic.append(person_map)

len(pic)

brain_map = third_parse.loc[:,['person_id','diagnosis_id']]
brain_map.count()

img = pd.Series( (v for v in pic) )
len(img)

brain_map = brain_map.reset_index(drop=True)

brain_map['map_img'] = img

brain_map.loc[0,:]



brain_map.to_pickle('./clint/docs/example_data/brain_map.pkz')



