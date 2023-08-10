import pandas as pd
pd.set_option('max_colwidth',100)
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
from sklearn import preprocessing as pp
import pickle
get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

couple_data = pd.read_pickle('./couple_data')

# See relationship for continuous features
plt.figure(figsize=(20,10))
couple_data_cont = couple_data.select_dtypes(include=['float32','float64','int64','int8'])
couple_data_cat = couple_data[[col for col in couple_data.columns if col not in couple_data_cont]]
corr = couple_data_cont.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr**2, mask=mask,annot=True)

# p value for highly correlated variables
print(sp.stats.pearsonr(couple_data['how_long_ago_first_met'], couple_data['how_long_relationship']))
print(sp.stats.pearsonr(couple_data['how_long_ago_first_met'], couple_data['how_long_ago_first_cohab']))
print(sp.stats.pearsonr(couple_data['how_long_ago_first_met'], couple_data['how_long_ago_first_romantic']))
print(sp.stats.pearsonr(couple_data['how_long_relationship'], couple_data['how_long_ago_first_romantic']))
print(sp.stats.pearsonr(couple_data['how_long_ago_first_cohab'], couple_data['how_long_ago_first_romantic']))
print(sp.stats.pearsonr(couple_data['how_long_ago_first_cohab'], couple_data['how_long_relationship']))

# P values are below 0.05, correlation between variables are statistically insignificant
# Do not need to remove them

couple_data_cat.info()

# income disparity between couples doesnt affect relationship outcome
sns.countplot(x=couple_data_cat['higher_income_earner'],hue=couple_data_cont['relationship_outcome_6yrs'])

# grow_up_same_city_town doesnt affect relationship outcome
plt.figure(figsize=(10,5))
sns.countplot(x=couple_data_cat['grow_up_same_city_town'],hue=couple_data_cont['relationship_outcome_6yrs'])

# parent_alive doesnt affect relationship outcome
plt.figure(figsize=(10,5))
sns.countplot(x=couple_data_cat['parent_alive'],hue=couple_data_cont['relationship_outcome_6yrs'])

# friend_intro_partner doesnt affect relationship outcome
plt.figure(figsize=(10,5))
sns.countplot(x=couple_data_cat['friend_intro_partner'],hue=couple_data_cont['relationship_outcome_6yrs'])

# self_intro_partner doesnt affect relationship outcome
plt.figure(figsize=(10,5))
sns.countplot(x=couple_data_cat['self_intro_partner'],hue=couple_data_cont['relationship_outcome_6yrs'])

# met_through_friends doesnt affect relationship outcome
sns.countplot(x=couple_data_cat['met_through_friends'],hue=couple_data_cont['relationship_outcome_6yrs'])

# married doesnt affect relationship outcome
sns.countplot(x=couple_data_cat['married'],hue=couple_data_cont['relationship_outcome_6yrs'])

# parental_approval doesnt affect relationship outcome
sns.countplot(x=couple_data_cat['parental_approval'],hue=couple_data_cont['relationship_outcome_6yrs'])

# whether parents are alive against relationship outcome
plt.figure(figsize=(10,5))
sns.countplot(x=couple_data_cat['parent_alive'],hue=couple_data_cont['relationship_outcome_6yrs'])

# What about years of education
cols = ['couple_yrsed_diff','couple_moms_yrsed_diff','relationship_outcome_6yrs']
couple_data_cont[cols].boxplot(by='relationship_outcome_6yrs', figsize=(20,10))

# Relationship seem to be due to imbalance data
# Resample Data for more accurate picture of relationships
# Convert categorical into dummies
import patsy
f = 'relationship_outcome_6yrs ~ ' + ' + '.join([col for col in couple_data.columns if col != 'relationship_outcome_6yrs']) + '-1'
y, X = patsy.dmatrices(f, data=couple_data, return_type='dataframe')

X.info()

X.to_pickle('./couple_data_without_resample_predictors')
y.to_pickle('./couple_data_without_resample_target')

y_orig = y

y = y.values.ravel()

# Resample using SMOTE + Tomek
from imblearn.combine import SMOTETomek
smtomek = SMOTETomek(random_state=42)
X_st, y_st = smtomek.fit_sample(X, y)

X_st_df = pd.DataFrame(X_st, columns=[col for col in X.columns])
y_st_df = pd.DataFrame(y_st, columns=['relationship_outcome_6yrs'])
# make sure dummy columns for categorical values are all 0/1
for cat_col in couple_data_cat.columns:
    for dum_col in X_st_df.columns:
        if dum_col.find(cat_col) >= 0:
            X_st_df[dum_col] = X_st_df[dum_col].astype('int64')

X_st_df.info()

X_st_df[['higher_income_earner[T.female_earn_more]']] = X_st_df[['higher_income_earner[T.female_earn_more]']].astype('int64')
X_st_df.age_difference = X_st_df.age_difference.astype('int64')
X_st_df.respondent_yrsed = X_st_df.respondent_yrsed.astype('int64')
X_st_df.partner_yrsed = X_st_df.partner_yrsed.astype('int64')
X_st_df.partner_mom_yrsed = X_st_df.partner_mom_yrsed.astype('int64')
X_st_df.respondent_mom_yrsed = X_st_df.respondent_mom_yrsed.astype('int64')

X_st_df.describe()

X_st_df.to_pickle('./couple_data_predictors')
y_st_df.to_pickle('./couple_data_target')

y_st_df.relationship_outcome_6yrs.value_counts()

y_orig.relationship_outcome_6yrs.value_counts()
503/float(1144)

# Get reverse imbalance dataset
merge_df = pd.concat([X_st_df,y_st_df],axis=1)
sub_df = merge_df[merge_df.relationship_outcome_6yrs == 0]
us_df = sub_df.sample(frac=0.44, random_state=42)
remerge_df = pd.concat([merge_df[merge_df.relationship_outcome_6yrs == 1], us_df], axis=0)

remerge_df.drop(labels=['relationship_outcome_6yrs'],axis=1).to_pickle('./couple_data_rev_imbal_predictors')
remerge_df.relationship_outcome_6yrs.to_pickle('./couple_data_rev_imbal_target')





