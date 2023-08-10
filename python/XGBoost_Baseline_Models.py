import pandas as pd

demographics = pd.read_csv('Demographics.csv')

ml = demographics.loc[:,'GENDER': 'HEART_DEATH_FLAG']
del ml['DOB']
del ml['DOD']
del ml['DOA']
ml.head()

# Get numerical data
ml_data = pd.get_dummies(ml, columns=['GENDER','ETHNICITY','MARITAL_STATUS', 'LANGUAGE', 'RELIGION', 'INSURANCE', 'ADMISSION_LOCATION'])
ml_data = ml_data[ml_data['OLD_FLAG']==0]

# Reduce population to only those with ages
ml_data = ml_data[ml_data['OLD_FLAG']==0]

# Produce output data sets to create models
heart_attacks = ml_data['HEART_ATTACK_FLAG']
athero_diagnosis = ml_data['ATHERO_DIAGNOSIS_FLAG']
deaths = ml_data['DEATH_FLAG']
heart_deaths = ml_data['HEART_DEATH_FLAG']

# Predict just deaths on non-diagnostic data
del ml_data['HEART_ATTACK_FLAG']
del ml_data['ATHERO_DIAGNOSIS_FLAG']
del ml_data['DEATH_FLAG']
del ml_data['HEART_DEATH_FLAG']
del ml_data['OLD_FLAG']
del ml_data['OUTSIDE_DEATH_FLAG']

ml_data.head()

# Create randomly undersampled data set
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(return_indices=True)
X_resampled, y_resampled, idx_resampled = rus.fit_sample(ml_data, deaths)

# Check sampling numbers
y_resampled = pd.Series(y_resampled)
print(y_resampled.value_counts())
print(deaths.value_counts())

# Create and test an XGBoost with 5 fold cross validation for predicting death on this model with no hyperparameter 
# optimization
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

base_model_XG = GradientBoostingClassifier()
scores = cross_val_score(base_model_XG, X_resampled, y_resampled, cv=5)
scores.mean()

# Same process for predicting heart attacks
rus = RandomUnderSampler(return_indices=True)
X_resampled, y_resampled, idx_resampled = rus.fit_sample(ml_data, heart_attacks)

# Check sampling numbers
y_resampled = pd.Series(y_resampled)
print(y_resampled.value_counts())
print(deaths.value_counts())

from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_sample(ml_data, heart_attacks)

y_resampled = pd.Series(y_resampled)
print(y_resampled.value_counts())

base_model_XG = GradientBoostingClassifier()
scores = cross_val_score(base_model_XG, X_resampled, y_resampled, cv=5)
scores.mean()

# First, define atherosclerosis diagnoses from non-atherosclerosis diagnoses
athero_pre = demographics[demographics['OLD_FLAG']==0]
athero_pos = athero_pre[athero_pre['ATHERO_DIAGNOSIS_FLAG']== 1]
athero_neg = athero_pre[athero_pre['ATHERO_DIAGNOSIS_FLAG']==0]

# Clean data sets
del athero_neg['CAUSE']
del athero_pos['CAUSE']

del athero_neg['ATHERO_DIAGNOSIS_FLAG']
del athero_pos['ATHERO_DIAGNOSIS_FLAG']

del athero_neg['OLD_FLAG']
del athero_pos['OLD_FLAG']

del athero_neg['OUTSIDE_DEATH_FLAG']
del athero_pos['OUTSIDE_DEATH_FLAG']

del athero_neg['SUBJECT_ID']
del athero_pos['SUBJECT_ID']

del athero_neg['DOB']
del athero_pos['DOB']

del athero_neg['DOD']
del athero_pos['DOD']

athero_pos['DOA']
del athero_pos['DOA']
del athero_neg['DOA']

athero_neg['HEART_ATTACK_FLAG']
del athero_neg['HEART_ATTACK_FLAG']
del athero_pos['HEART_ATTACK_FLAG']

del athero_pos['Unnamed: 0']

len(athero_pos)

athero_pos.head()

# Create Outcome data sets
athero_heartdeath = pd.Series(athero_pos['HEART_DEATH_FLAG'])
athero_death = pd.Series(athero_pos['DEATH_FLAG'])

del athero_pos['HEART_DEATH_FLAG']
del athero_pos['DEATH_FLAG']

# Get dummies
athero_pos = pd.get_dummies(athero_pos, columns=['GENDER','ETHNICITY','MARITAL_STATUS', 'LANGUAGE', 'RELIGION', 'INSURANCE', 'ADMISSION_LOCATION'])

# Check outcome numbers
print(athero_heartdeath.value_counts())
print(athero_death.value_counts())

base_model_XG = GradientBoostingClassifier()
scores = cross_val_score(base_model_XG, athero_pos, athero_death, cv=5)
scores.mean()

athero_pos.head()



