import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
from sklearn import preprocessing as pp
get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

# Features are mainly demographic data from onset of survey,
# excluding 1 relationship centric feature - 'relationship quality'

# Goal is to only use data gathered at the start of the survey, Wave 1,
# to try to predict the outcome of partnered respondents, based on respondent and partner demographic data
# This allows us to investigate and discover what type of individuals 
# have better chance of surviving a 6 year relationship

df_w5 = pd.read_stata('./datasets/HCMST_wave_5_supplement_ver_1.dta')
df_w4 = pd.read_stata('./datasets/wave_4_supplement_v1_2.dta')
df_w123 = pd.read_stata('./datasets/HCMST_ver_3.04.dta')

w5_cols = df_w5.columns.tolist()
w4_cols = df_w4.columns.tolist()
w123_cols = df_w123.columns.tolist()

# Checking number of features for each wave
print(len(w5_cols), len(w4_cols), len(w123_cols))

# checking rows and columns for each wave
print(df_w123.shape, df_w4.shape, df_w5.shape)

# 
wave_1_couples = df_w123.loc[(df_w123['qflag'] == 'partnered') & (df_w123['partner_deceased'] == 'not deceased'), 'caseid_new':]

wave_1_couples.info()

# Get relationship status of couples that have deceased partners
w2_dead = wave_1_couples[wave_1_couples.w2_broke_up == 'partner passed away'].index
w3_dead = wave_1_couples[wave_1_couples.w3_broke_up == 'partner deceased'].index
w4_dead = df_w4[df_w4.w4_broke_up == 'partner passed away'].index
w5_dead = df_w5[df_w5.w5_broke_up == 'partner deceased'].index

print(len(wave_1_couples.loc[wave_1_couples.loc[w2_dead,['qflag']].index][wave_1_couples.qflag == 'partnered'].index))
print('')
print(len(wave_1_couples.loc[wave_1_couples.loc[w3_dead,['w2_broke_up']].index][wave_1_couples.w2_broke_up == 'still together'].index))
print('') 
print(len(wave_1_couples.loc[wave_1_couples.loc[w4_dead,['w3_broke_up']].index][wave_1_couples.w3_broke_up == 'still together'].index))
print('') 
print(len(df_w4.loc[df_w4.loc[w5_dead,['w4_broke_up']].index][df_w4.w4_broke_up == 'still together'].index))

w2_dead_st = wave_1_couples.loc[wave_1_couples.loc[w2_dead,['qflag','caseid_new']].index][wave_1_couples.qflag == 'partnered'].caseid_new.values
w3_dead_st = wave_1_couples.loc[wave_1_couples.loc[w3_dead,['w2_broke_up','caseid_new']].index][wave_1_couples.w2_broke_up == 'still together'].caseid_new.values
w4_dead_st = wave_1_couples.loc[wave_1_couples.loc[w4_dead,['w3_broke_up','caseid_new']].index][wave_1_couples.w3_broke_up == 'still together'].caseid_new.values
w5_dead_st = df_w4.loc[df_w4.loc[w5_dead,['w4_broke_up','caseid_new']].index][df_w4.w4_broke_up == 'still together'].caseid_new.values

# prior relationship status of all couples that have deceased partners is still together
tot_dead_list = np.concatenate((w2_dead_st,w3_dead_st,w4_dead_st,w5_dead_st))

wave_2_couples_broke_up = wave_1_couples.loc[wave_1_couples['w2_broke_up']=='broke up']
wave_3_couples_broke_up = wave_1_couples.loc[wave_1_couples['w3_broke_up']=='broke up']
w4_breakup = df_w4.loc[df_w4['w4_broke_up']=='broke up']
w5_breakup = df_w5.loc[df_w5['w5_broke_up']=='broke up']
survived_couples=df_w5.loc[df_w5['w5_broke_up']=='still together']

w2_breakup_list=wave_2_couples_broke_up['caseid_new'].tolist()
w3_breakup_list=wave_3_couples_broke_up['caseid_new'].tolist()
w4_breakup_list=w4_breakup['caseid_new'].tolist()
w5_breakup_list=w5_breakup['caseid_new'].tolist()
survived_couples_list=survived_couples['caseid_new'].tolist()

# Must make sure all couples in dataset are partnered right from the start
cp_index = np.concatenate((w4_breakup.index,w5_breakup.index,survived_couples.index))
wave_1_couples.loc[cp_index].head()

# for x in tot_dead_st:
#     if x in w2_breakup_list or x in w3_breakup_list or x in w4_breakup_list or x in w5_breakup_list:
#         print(x)

# check the imbalance of the data
print(len(w5_breakup_list)+len(w4_breakup_list)+len(w3_breakup_list)+len(w2_breakup_list))
print(len(tot_dead_list)+len(survived_couples_list))

breakup_list = np.concatenate((w5_breakup_list,w4_breakup_list,w3_breakup_list,w2_breakup_list), axis=0)
together_list = np.concatenate((tot_dead_list,survived_couples_list), axis=0)

# only want wave 1 features
features = wave_1_couples.loc[:,:'coresident'].columns
couple_data = wave_1_couples[features]

couple_data.set_index(keys=['caseid_new'], inplace=True)

# dropping obviously irrelevant columns
cols_to_drop = couple_data.loc[w2_breakup_list,'pphhcomp11_member2_age':'weight_couples_coresident'].columns

couple_data.drop(labels=cols_to_drop, axis=1, inplace=True)

couple_breakup = couple_data.loc[breakup_list,:]
couple_together = couple_data.loc[together_list,:]

print(couple_breakup.shape)
print(couple_together.shape)

couple_breakup['relationship_outcome_6yrs'] = [1 for x in range(couple_breakup.shape[0])]
couple_together['relationship_outcome_6yrs'] = [0 for x in range(couple_together.shape[0])]

print(couple_breakup.relationship_outcome_6yrs.unique())
print(couple_together.relationship_outcome_6yrs.unique())

couple_data = pd.concat([couple_breakup, couple_together], axis=0)

# drop more unnecessary columns that will not be useful features
couple_data.drop(labels=['weight1','weight2'], axis=1, inplace=True)
# drop age category columns, already have age continuous column
couple_data.drop(labels=['ppagecat','ppagect4'], axis=1, inplace=True)
# use education category instead of education
# coresident and how long cohab is the same, remove coresident
couple_data.drop(labels=['ppeduc','coresident'], axis=1, inplace=True)
# use continuous household income instead of categorical household income
couple_data.drop(labels=['ppincimp'], axis=1, inplace=True)
# Drop MSA metropolitan statistical area, irrelevancy
couple_data.drop(labels=['ppmsacat'], axis=1, inplace=True)
# drop ppt01, ppt1317, ppt25, ppt612, features aggregated in children_in_hh
couple_data.drop(labels=['ppt01','ppt1317','ppt25','ppt612'], axis=1, inplace=True)
# drop ppq14arace, redundant column, there are individual race columns
couple_data.drop(labels=['ppq14arace'], axis=1, inplace=True)
# drop ppppcmdate_yrmo, pppadate_yrmo, date of survey is irrelevant
couple_data.drop(labels=['ppppcmdate_yrmo','pppadate_yrmo'], axis=1, inplace=True)
# drop HCMST_main_interview_yrmo, date of interview is irrelevant
# drop interview duration, qflag - all partnered
couple_data.drop(labels=['HCMST_main_interview_yrmo','duration','qflag'], axis=1, inplace=True)
# drop papglb_status, same as glbstatus
# drop recsource, source of recruitment is irrelevant
# drop s1a,s2,q3_codes,q5,q15a1_compressed,q17c,q17d redundant or too many NAs, 
couple_data.drop(labels=['glbstatus','papglb_status','recsource','s1','s1a','s2','q3_codes','q5','q15a1_compressed','q17c','q17d'], axis=1, inplace=True)
# drop q18a_1, q18a_2, q18a_3, q18b_codes, q18c_codes, low variance and too many NAs
couple_data.drop(labels=['q18a_1','q18a_2','q18a_3','q18b_codes','q18c_codes','q18a_refused'], axis=1, inplace=True)
# drop q20, q21a_refusal, q21b_refusal,q21c_refusal,q21d_refusal,q21e,q21e_refusal,q24_codes
# 'q31_9','q31_other_text_entered','q33_7','q33_other_text_entered',not usable column
# q35_codes, q35_text_entered, summary_q24_total
couple_data.drop(labels=['q20','q21a_refusal','q21b_refusal','q21c_refusal','q21d_refusal',
                        'q21e','q21e_refusal','q24_codes','q31_9','q31_other_text_entered',
                        'q33_7','q33_other_text_entered','q35_codes','q35_text_entered',
                        'summary_q24_total'], axis=1, inplace=True)
# drop marrynotreally, marrycountry, civilnotreally, partner_deceased
# partner_religion_reclassified, partner_religion_child_reclass, own_religion_child_reclass
# q32_internet, how_met_online, potential_partner_gender_recodes, how_long_ago_first_met_cat
# duplicated representation of previous columns, too many NA
couple_data.drop(labels=['marrynotreally','marrycountry','civilnotreally','partner_deceased',
                        'partner_religion_reclassified','partner_religion_child_reclass',
                        'own_religion_child_reclass','q32_internet','how_met_online',
                        'either_internet','either_internet_adjusted','potential_partner_gender_recodes',
                        'how_long_ago_first_met_cat','pphouseholdsize'], axis=1, inplace=True)
# drop q24_R_friend, q24_P_friend, q24_R_family, q24_P_family, q24_R_neighbor, q24_P_neighbor
# q24_R_cowork, q24_P_cowork,papreligion,q13b,respondent_religion_at_16,partner_religion_at_16
# q7b, q8b,q30
# columns are aggregated into other columns
couple_data.drop(labels=['q24_R_friend','q24_P_friend','q24_R_family','q24_P_family','q24_R_neighbor','q24_P_neighbor',
                        'q24_R_cowork','q24_P_cowork','papreligion','q13b','respondent_religion_at_16',
                        'partner_religion_at_16','q7b','q8b','q30'], axis=1, inplace=True)

# Remove more duplicate columns
couple_data.drop(labels=['q31_4','q32','q31_2','q31_3','q31_8','q31_6'], axis=1, inplace=True)
# drop q17a, use q17b instead, current marriage already taken into account in 'married' column
couple_data.drop(labels=['q17a'], axis=1, inplace=True)

# remove identical columns based on data dictionary
couple_data.drop(labels=['q4','q6a','q6b','q8a','q9','q10','q11','q13a','q14','q19','q21a','q21b','q21c'], axis=1, inplace=True)

# Drop relationship_quality as feature will carry most weight in the prediction
# We also will want more objective than subjective features
couple_data.drop(labels=['relationship_quality'], axis=1, inplace=True)
# remove q34, same as relationship quality
# remove ppmarit, already have married or not column
# remove ppeducat, already have years of education
# remove ppage, already have age difference
# remove children in household, not very informative as data is concentrated on 0 children
# remove ppethm, already have respondent / partner race
# remove number of adults in household, data too concentrated on smaller numbers
# remove ppgender, redundant as we only need to know whether couple is same sex or not
# remove ppreg4 and ppreg9, columns only pertains to USA
# remove gender attraction, redundant column as already have same sex couple columns
# remove alt partner gender, already have same sex couple column

couple_data.drop(labels=['q34','ppmarit','ppeducat','ppage','children_in_hh','ppethm','pphispan',                         'pprace_white','pprace_black','pprace_someotherrace','ppt18ov',                         'ppreg4','ppreg9',                         'papglb_friend','gender_attraction',                         'alt_partner_gender'], axis=1, inplace=True)

couple_data.info()

# Iterate through each series in the dataframe, impute nulls with highest mode, binary values
import random
sr = random.SystemRandom()

def impute_bin_cols(cols):
    
    for col in cols:
        # get number of keys
        key_arr = [k for k in couple_data[col].value_counts().sort_values(ascending=False).to_dict().keys()]
        greater_key = key_arr[0]
        greater_counts = couple_data[col].value_counts().to_dict()[greater_key]

        if len(couple_data[col].value_counts().sort_values(ascending=False).to_dict().keys()) > 1:
            lesser_key = key_arr[1]
            lesser_counts = couple_data[col].value_counts().to_dict()[lesser_key]

            if greater_counts > lesser_counts:
                couple_data[col].fillna(greater_key, inplace=True)
            else:
                couple_data[col].fillna(sr.choice([greater_key,lesser_key]), inplace=True)
        else:
            couple_data[col].fillna(greater_key, inplace=True)
    
    return couple_data[cols].isnull().sum()

couple_data.isnull().sum().sort_values(ascending=False)

# investigate null features
# drop features with more than 70% null
couple_data.drop(labels=['q22'], axis=1, inplace=True)
# drop home_country_recode, too many NAs
couple_data.drop(labels=['home_country_recode'], axis=1, inplace=True)

# fill null and 'refused' values
couple_data.q17b.fillna('never married', inplace=True)
couple_data.q17b = couple_data.q17b.map(lambda x: 'never married' if x == 'refused' else x)

couple_data.q26 = couple_data.q26.map(lambda x: 'did not attend same college or university' if x == 'refused' else x)

couple_data.parental_approval.fillna("don't approve or don't know", inplace=True)
# fill null values for continuous variable
couple_data.q21d.fillna(couple_data.q21d.median(), inplace=True)
couple_data.how_long_ago_first_cohab.fillna(couple_data.how_long_ago_first_cohab.median(), inplace=True)
# partner mum years of education
couple_data.partner_mom_yrsed.fillna(couple_data.partner_mom_yrsed.median(), inplace=True)
# distancemoved_10mi
couple_data.distancemoved_10mi.fillna(couple_data.distancemoved_10mi.median(), inplace=True)
# how_long_ago_first_romantic
couple_data.how_long_ago_first_romantic.fillna(couple_data.how_long_ago_first_romantic.median(), inplace=True)
# how_long_relationship
couple_data.how_long_relationship.fillna(couple_data.how_long_relationship.median(), inplace=True)
# respondent_mom_yrsed
couple_data.respondent_mom_yrsed.fillna(couple_data.respondent_mom_yrsed.median(), inplace=True)
# how_long_ago_first_met
couple_data.how_long_ago_first_met.fillna(couple_data.how_long_ago_first_met.median(), inplace=True)
# q16 how many of your relatives do you see in person at least once a month
couple_data.q16 = couple_data.q16.astype('float64')
couple_data.q16.fillna(np.median([int(i) for i in couple_data.q16.unique() if np.isnan(i) == False]), inplace=True)
# age difference
couple_data.age_difference.fillna(couple_data.age_difference.median(), inplace=True)# q9 how old is your partner
# respondent race 
couple_data.respondent_race.fillna('NH white', inplace=True)
# partner religion at 16 years old
couple_data.partner_relig_16_cat.fillna('No religion', inplace=True)
# respondent religion at 16 years old
couple_data.respondent_relig_16_cat.fillna('No religion', inplace=True)
# partner years of education
couple_data.partner_yrsed.fillna(couple_data.partner_yrsed.median(), inplace=True)
# partner race
couple_data.partner_race.fillna('NH white', inplace=True)

# q24 columns 
cols = [x for x in couple_data.columns if 'q24_' in x]
impute_bin_cols(cols)

# more binary columns
cols = ['met_through_as_coworkers','met_through_friends','met_through_family',
        'papevangelical','met_through_as_neighbors','US_raised']
impute_bin_cols(cols)

# loop through every feature and check values again
for col in couple_data.columns:
    if couple_data[col].dtype.name == 'category' or couple_data[col].dtype.name == 'object':
        print(couple_data[col].value_counts())
        print('------------------------')

cols = [col for col in couple_data.columns if 'pprace' in col]
# Convert all refused value into Nan
for col in cols:
    # get hi
    couple_data[col] = couple_data[col].map(lambda x: np.nan if x == 'refused' else x)
# Impute them with larger count
impute_bin_cols(cols)

# remove ambiguous from q7a, q8a, q13a,q19,q25,q27,q28,q31_1,q31_2,q31_3,q31_4,q31_5,q31_6,q31_7,q31_8
# q33_1,q33_2,q33_3
cols = ['q7a','q25','q27','q28','q31_1','q31_5','q31_7','q33_1','q33_2','q33_3','q33_4','q33_5','q33_6']
for col in cols:
    couple_data[col] = couple_data[col].map(lambda x: np.nan if x == 'refused' else x)
impute_bin_cols(cols)

# democrat                         716
# republican                       404
# independent                      262
# no preference                    238
# another party, please specify     22
# refused                            5
# Name: q12, dtype: int64
couple_data.q12 = couple_data.q12.map(lambda x: 'no preference' if x == 'refused' else x)

# i earned more                      757
# partner earned more                669
# we earned about the same amount    211
# refused                             10
# Name: q23, dtype: int64
couple_data.q23 = couple_data.q23.map(lambda x: 'we earned about the same amount' if x == 'refused' else x)

# father and mother                      787
# neither father nor mother are alive    441
# mother only                            334
# father only                             78
# refused                                  7
# Name: q29, dtype: int64
couple_data.q29 = couple_data.q29.map(lambda x: 'father and mother' if x == 'refused' else x)

# check features with 1 distinct value more than or equals to 90% of the sample
# if feature important, seek workaround, otherwise remove
def get_low_var(df):
    low_var_cols = []
    cols = df.columns
    
    for col in cols:
        arr = np.array(df[col].value_counts() / df.shape[0])
        for prop in arr:
            if prop >= 0.9:
                low_var_cols.append(col)
                
    return low_var_cols

# remove features with 1 distinct value more than or equals to 95% of sample
low_var_features = get_low_var(couple_data)
couple_data.drop(labels=low_var_features, axis=1, inplace=True)

# rename question columns
couple_data.rename(index=str, columns={'q4':'partner_gender','q7a':'partner_christ_type',
                                      'q12':'partner_politic_view','q16':'relatives_seen_per_month',
                                      'q17b':'marriage_count','q21d':'age_when_married',
                                       'q23':'higher_income_earner','q25':'same_high_school',
                                      'q26':'same_college_uni','q27':'grow_up_same_city_town',
                                      'q28':'both_parents_knew_before_met','q29':'parent_alive',
                                      'q31_1':'met_partner_work','q33_1':'fam_intro_partner',
                                      'q33_2':'friend_intro_partner','q33_3':'colleague_intro_partner',
                                      'q33_6':'self_intro_partner'}, inplace=True)

# Group values in ppwork into employee, self-employed, not-working 
couple_data.ppwork = couple_data.ppwork.map(lambda x: 'employee' if x.find('employee') >= 0 else                                            'self-employed' if x.find('self-employed') >= 0 else                                            'not-working')

couple_data.info(verbose=False)

# Include gender in higher income earner column to make it more informative
couple_income = couple_data.higher_income_earner.astype('object') + '_' + couple_data.ppgender.astype('object')
couple_data.higher_income_earner = couple_income.map(lambda x: 'male_earn_more' if x.find('earned more_male') >= 0 else                  'female_earn_more' if x.find('earned more_female') >= 0 else 'both_earn_same')

# Couple political views combination
# Make sure unique values for both are the same
couple_data.partner_politic_view = couple_data.partner_politic_view.map(lambda x: 'other' if x != 'democrat' and x != 'republican' else x)
couple_political_view_comb = couple_data.partner_politic_view.astype('object') + '_' + couple_data.pppartyid3.astype('object') 

couple_political_view_comb.value_counts()

# Remove Duplicates
couple_data['couple_politic_view_comb'] = couple_political_view_comb.map(lambda x: 'democrat_other' if x.find('democrat') >= 0 and x.find('other') >= 0                                 else 'republican_other' if x.find('republican') >= 0 and x.find('other') >= 0                                 else 'democrat_republican' if x.find('democrat') >= 0 and x.find('republican') >= 0                                 else x)

# remove individual political view columns
couple_data.drop(labels=['partner_politic_view','pppartyid3'], axis=1, inplace=True)

# couple evangelical / born again or not, combinations
couple_data.partner_christ_type = couple_data.partner_christ_type.map(lambda x: 'evang or born again' if x == 'yes' else x)
couple_data.papevangelical = couple_data.papevangelical.map(lambda x: 'evang or born again' if x == 'yes' else x)

couple_evang_comb = couple_data.partner_christ_type.astype('object') + '_' + couple_data.papevangelical.astype('object')
couple_data['couple_evang_comb'] = couple_evang_comb.map(lambda x: 'no_evang or born again' if x.find('evang or born again') >= 0 and x.find('no') >= 0 else x)

# remove individual evang columns
couple_data.drop(labels=['partner_christ_type','papevangelical'], axis=1, inplace=True)

# couples are assumed to have not changed their religion
# group no religion, jewish and neiether christian nor jewish together as other
couple_data.respondent_relig_16_cat = couple_data.respondent_relig_16_cat.map(lambda x: 'other' if x != 'Protestant or oth Christian' and                                        x != 'Catholic' else x)
couple_data.partner_relig_16_cat = couple_data.partner_relig_16_cat.map(lambda x: 'other' if x != 'Protestant or oth Christian' and                                        x != 'Catholic' else x)

couple_relig_comb = couple_data.respondent_relig_16_cat.astype('object') + '_' + couple_data.partner_relig_16_cat.astype('object')
couple_data['couple_relig_comb'] = couple_relig_comb.map(lambda x: 'Protestant or oth Christian_Catholic' if x.find('Protestant or oth Christian') >= 0 and                      x.find('Catholic') >= 0 else 'Protestant or oth Christian_other' if x.find('Protestant or oth Christian') >= 0 and                      x.find('other') >= 0 else 'Catholic_other' if x.find('Catholic') >= 0 and x.find('other') >= 0                       else x)

# remove individual religion columns
couple_data.drop(labels=['respondent_relig_16_cat','partner_relig_16_cat'], axis=1, inplace=True)

# it seems like age when married column does not tally with married or not column,
# since couple is not married, there should not be an age when they were married
couple_data[couple_data.married == 'not married'].loc[:,['age_when_married']].age_when_married.value_counts()

# dropping age when married column
couple_data.drop(labels=['age_when_married'], axis=1, inplace=True)

# trim leading and trailing white spaces
couple_data.respondent_race = couple_data.respondent_race.map(lambda x: x.strip())
couple_data.partner_race = couple_data.partner_race.map(lambda x: x.strip())
print(couple_data.respondent_race.value_counts())
print(couple_data.partner_race.value_counts())

# To make race columns more useful, we will reduce it to just two unique values, NH white & others
couple_data.respondent_race = couple_data.respondent_race.map(lambda x: 'other' if x != 'NH white' else x)
couple_data.partner_race = couple_data.partner_race.map(lambda x: 'other' if x != 'NH white' else x)

# Get couple race combinations
couple_race_comb = couple_data.respondent_race.astype('object') + '_' + couple_data.partner_race.astype('object')
couple_data['couple_race_comb'] = couple_race_comb.map(lambda x: 'NH white_other' if x.find('NH white') >= 0 and x.find('other') >= 0 else x)

# Drop individual race columns
couple_data.drop(labels=['respondent_race','partner_race'], axis=1, inplace=True)

# Combine ppwork with gender to make it more insightful
gender_work_status = couple_data.ppwork.astype('object') + '_' + couple_data.ppgender.astype('object')
# gender_work_status.value_counts()
couple_data['gender_work_status'] = gender_work_status

# Drop ppwork column
couple_data.drop(labels=['ppwork'], axis=1, inplace=True)

couple_data.pphouse.value_counts()

couple_data.pphouse = couple_data.pphouse.map(lambda x: 'a mobile home' if x == 'boat, rv, van, etc.' else x)

# combine ppnet and gender to make it more informative
gender_internet_access = couple_data.ppnet.astype('object') + '_' + couple_data.ppgender.astype('object')
# gender_internet_access.value_counts()
couple_data['gender_internet_access'] = gender_internet_access

couple_data.drop(labels=['ppnet'], axis=1, inplace=True)

# combine household head and gender to make it more informative
hhhead_gender = couple_data.pphhhead.astype('object') + '_' + couple_data.ppgender.astype('object')
hhhead_gender.value_counts()

# no and female = yes male, vice versa
hhhead_gender = hhhead_gender.map(lambda x: 'yes_male' if x == 'no_female' else 'yes_female' if x == 'no_male' else x)
couple_data['hhhead_gender'] = hhhead_gender

# drop pphhhead
couple_data.drop(labels=['pphhhead'], axis=1, inplace=True)

# convert how many times individual has been married to binary, never married before & married before
couple_data.marriage_count.value_counts()

couple_data.marriage_count = couple_data.marriage_count.map(lambda x: 'married before' if x != 'never married' else x)

marriage_count_gender = couple_data.marriage_count.astype('object') + '_' + couple_data.ppgender.astype('object')
couple_data['marriage_count_gender'] = marriage_count_gender

# drop marriage count, only pertains to individual
couple_data.drop(labels=['marriage_count'], axis=1, inplace=True)

hhinc_median = couple_data.hhinc.median()
couple_data.hhinc = couple_data.hhinc.map(lambda x: 'high_hhinc' if x > hhinc_median else 'low_hhinc')

hhinc_median

# convert rent into for cash or not
couple_data.pprent.value_counts()

couple_data.pprent = couple_data.pprent.map(lambda x: 'free' if x != 'rented for cash' else x)

# difference in years of education between partners
couple_data['couple_yrsed_diff'] = np.abs(couple_data.partner_yrsed - couple_data.respondent_yrsed)

couple_data.drop(labels=['partner_yrsed','respondent_yrsed'], axis=1, inplace=True)

# difference in years of education between partner's mums
couple_data['couple_moms_yrsed_diff'] = np.abs(couple_data.partner_mom_yrsed - couple_data.respondent_mom_yrsed)

couple_data.drop(labels=['partner_mom_yrsed','respondent_mom_yrsed'], axis=1, inplace=True)

# drop ppgender, only pertains to individual
couple_data.drop(labels=['ppgender'], axis=1, inplace=True)

# Create new feature called partner to respondent years of education
couple_data.info(verbose=False)

# pickle cleaned data
couple_data.to_pickle('./couple_data')



