import pandas as pd
import matplotlib.pyplot as plt
import requests
import urllib
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import numpy as np
get_ipython().magic('matplotlib inline')
import dateutil.parser

#H-2A Data
df_H2a_2015 = pd.read_excel('https://www.foreignlaborcert.doleta.gov/docs/py2015q4/H-2A_Disclosure_Data_FY15_Q4.xlsx')
df_H2a_2014 = pd.read_excel('https://www.foreignlaborcert.doleta.gov/docs/py2014q4/H-2A_FY14_Q4.xlsx')
df_H2a_2013 = pd.read_excel('https://www.foreignlaborcert.doleta.gov/docs/h_2a/H2A_FY2013.xls')
df_H2a_2012 = pd.read_excel('https://www.foreignlaborcert.doleta.gov/docs/h_2a/H-2A_FY2012.xlsx')
df_H2a_2011 = pd.read_excel('https://www.foreignlaborcert.doleta.gov/docs/h_2a/H-2A_FY2011.xlsx')
df_H2a_2010 = pd.read_excel('https://www.foreignlaborcert.doleta.gov/docs/h_2a/H-2A_FY2010.xlsx')
df_H2a_2009 = pd.read_excel('https://www.foreignlaborcert.doleta.gov/docs/h_2a/H2A_FY2009.xlsx')

#H-2B Data
df_H2b_2015 = pd.read_excel('https://www.foreignlaborcert.doleta.gov/docs/py2015q4/H-2B_Disclosure_Data_FY15_Q4.xlsx')
df_H2b_2014 = pd.read_excel('https://www.foreignlaborcert.doleta.gov/docs/py2014q4/H-2B_FY14_Q4.xlsx')
df_H2b_2013 = pd.read_excel('https://www.foreignlaborcert.doleta.gov/docs/h_2b/H-2B_FY2013.xls')
df_H2b_2012 = pd.read_excel('https://www.foreignlaborcert.doleta.gov/docs/h_2b/H-2B_FY2012.xlsx')
df_H2b_2011 = pd.read_excel('https://www.foreignlaborcert.doleta.gov/docs/h_2b/H-2B_FY2011.xlsx')
df_H2b_2010 = pd.read_excel('https://www.foreignlaborcert.doleta.gov/docs/h_2b/H-2B_FY2010.xlsx')
df_H2b_2009 = pd.read_excel('https://www.foreignlaborcert.doleta.gov/docs/h_2b/H-2B_FY2009.xlsx')

#H-1B Data
df_H1b_2015 = pd.read_excel('https://www.foreignlaborcert.doleta.gov/docs/py2015q4/H-1B_Disclosure_Data_FY15_Q4.xlsx')
df_H1b_2014 = pd.read_excel('https://www.foreignlaborcert.doleta.gov/docs/py2014q4/H-1B_FY14_Q4.xlsx')
df_H1b_2013 = pd.read_excel('https://www.foreignlaborcert.doleta.gov/docs/lca/LCA_FY2013.xlsx')
df_H1b_2012 = pd.read_excel('https://www.foreignlaborcert.doleta.gov/docs/py2012_q4/LCA_FY2012_Q4.xlsx')
df_H1b_2011 = pd.read_excel('https://www.foreignlaborcert.doleta.gov/docs/lca/H-1B_iCert_LCA_FY2011_Q4.xlsx')
df_H1b_2010 = pd.read_excel('https://www.foreignlaborcert.doleta.gov/docs/lca/H-1B_FY2010.xlsx')
df_H1b_2009 = pd.read_excel('https://www.foreignlaborcert.doleta.gov/docs/lca/Icert_%20LCA_%20FY2009.xlsx')
df_H1b_2009_2 = pd.read_excel('https://www.foreignlaborcert.doleta.gov/docs/lca/H-1B_Case_Data_FY2009.xlsx')

#Pulling out relevant columns H2B
df_H2b_2015_ = df_H2b_2015[["DECISION_DATE", "CASE_NUMBER", "CASE_STATUS", "EMPLOYER_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", "PREVAILING_WAGE", "BASIC_RATE_OF_PAY", 'WORKSITE_STATE', 'JOB_TITLE', 'SOC_TITLE', "VISA_CLASS"]].copy()
df_H2b_2014_ = df_H2b_2014[["RECENT_DECISION_DATE", "CASE_NUMBER", "CASE_STATUS", "ALIEN_WORK_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", "PREVIALING_WAGE", "BASIC_RATE_OF_PAY", 'ALIEN_WORK_STATE', 'JOB_TITLE', 'SOC_NAME',"VISA_CLASS"]].copy()
df_H2b_2013_ = df_H2b_2013[["DECISION_DATE", "CASE_NUMBER", "CASE_STATUS", "ALIEN_WORK_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", "PREVIALING_WAGE", "BASIC_RATE_OF_PAY", 'ALIEN_WORK_STATE', 'JOB_TITLE', 'SOC_NAME',"VISA_CLASS"]].copy()
df_H2b_2012_ = df_H2b_2012[["DECISION_DATE", "CASE_NO", "CASE_STATUS", "ALIEN_WORK_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", "PREVAILING_WAGE", "BASIC_RATE_OF_PAY", 'ALIEN_WORK_STATE', 'JOB_TITLE', 'SOC_NAME',"VISA_CLASS"]].copy()
df_H2b_2011_ = df_H2b_2011[["DECISION_DATE", "CASE_NO", "CASE_STATUS", "ALIEN_WORK_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", "PREVAILING_WAGE", "BASIC_RATE_OF_PAY", 'ALIEN_WORK_STATE', 'JOB_TITLE', 'SOC_NAME',"VISA_CLASS"]].copy()
df_H2b_2010_ = df_H2b_2010[["DECISION_DATE", "CASE_NO", "CASE_STATUS", "ALIEN_WORK_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", "PREVAILING_WAGE", "BASIC_RATE_OF_PAY", 'ALIEN_WORK_STATE', 'JOB_TITLE', 'DOT_OCCUPATIONAL_CODE',"APPLICATION_TYPE"]].copy()
df_H2b_2009_ = df_H2b_2009[["DECISION_DATE", "CASE_NO", "CASE_STATUS", "ALIEN_WORK_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", "PREVAILING_WAGE", "BASIC_RATE_OF_PAY", 'ALIEN_WORK_STATE', 'JOB_TITLE']].copy()

#Pulling out relevant columns H2A
df_H2a_2015_ = df_H2a_2015[["DECISION_DATE", "CASE_NUMBER", "CASE_STATUS", "EMPLOYER_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", "BASIC_UNIT_OF_PAY", "BASIC_RATE_OF_PAY", 'WORKSITE_STATE', 'JOB_TITLE', 'SOC_TITLE', "VISA_CLASS"]].copy()
df_H2a_2014_ = df_H2a_2014[["DECISION_DATE", "CASE_NO", "CASE_STATUS", "EMPLOYER_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", "BASIC_UNIT_OF_PAY", "BASIC_RATE_OF_PAY", 'WORKSITE_LOCATION_STATE', 'JOB_TITLE', 'SOC_TITLE', "VISA_CLASS"]].copy()
df_H2a_2013_ = df_H2a_2013[["DECISION_DATE", "CASE_NO", "CASE_STATUS", "EMPLOYER_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", "BASIC_UNIT_OF_PAY", "BASIC_RATE_OF_PAY", 'ALIEN_WORK_STATE', 'JOB_TITLE', "VISA_CLASS"]].copy()
df_H2a_2012_ = df_H2a_2012[["DECISION_DATE", "CASE_NO", "CASE_STATUS", "EMPLOYER_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", "BASIC_UNIT_OF_PAY", "BASIC_RATE_OF_PAY", 'ALIEN_WORK_STATE', 'JOB_TITLE', "VISA_CLASS"]].copy()
df_H2a_2011_ = df_H2a_2011[["DECISION_DATE", "CASE_NO", "CASE_STATUS", "EMPLOYER_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", "BASIC_UNIT_OF_PAY", "BASIC_RATE_OF_PAY", 'ALIEN_WORK_STATE', 'JOB_TITLE', "APPLICATION_TYPE"]].copy()
df_H2a_2010_ = df_H2a_2010[["DECISION_DATE", "CASE_NO", "CASE_STATUS", "EMPLOYER_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", "BASIC_UNIT_OF_PAY", "BASIC_RATE_OF_PAY", 'ALIEN_WORK_STATE', 'JOB_TITLE',]].copy()
df_H2a_2009_ = df_H2a_2009[["DECISION_DATE", "CASE_NO", "CASE_STATUS", "EMPLOYER_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", "BASIC_UNIT_OF_PAY", "BASIC_RATE_OF_PAY", 'ALIEN_WORK_STATE', 'JOB_TITLE',]].copy()

#Pulling out relevant columns H1B
df_H1b_2015_ = df_H1b_2015[['DECISION_DATE', 'CASE_NUMBER', "CASE_STATUS", "EMPLOYER_STATE", 'EMPLOYER_NAME', 'TOTAL WORKERS', 'WORKSITE_STATE', 'JOB_TITLE', 'SOC_NAME', "VISA_CLASS"]].copy()
df_H1b_2014_ = df_H1b_2014[['DECISION_DATE', 'LCA_CASE_NUMBER', 'STATUS', 'LCA_CASE_EMPLOYER_STATE', 'LCA_CASE_EMPLOYER_NAME', 'TOTAL_WORKERS', 'LCA_CASE_WORKLOC1_STATE', 'LCA_CASE_JOB_TITLE', 'LCA_CASE_SOC_NAME', 'VISA_CLASS']].copy()
df_H1b_2013_ = df_H1b_2013[['Decision_Date', 'LCA_CASE_NUMBER', 'STATUS', 'LCA_CASE_EMPLOYER_STATE', 'LCA_CASE_EMPLOYER_NAME', 'TOTAL_WORKERS', 'LCA_CASE_WORKLOC1_STATE','LCA_CASE_JOB_TITLE', 'LCA_CASE_SOC_NAME', 'VISA_CLASS']].copy()
df_H1b_2012_ = df_H1b_2012[['DECISION_DATE', 'LCA_CASE_NUMBER', 'STATUS', 'LCA_CASE_EMPLOYER_STATE', 'LCA_CASE_EMPLOYER_NAME', 'TOTAL_WORKERS', 'LCA_CASE_WORKLOC1_STATE','LCA_CASE_JOB_TITLE', 'LCA_CASE_SOC_NAME', 'VISA_CLASS']].copy()
df_H1b_2011_ = df_H1b_2011[['DECISION_DATE', 'LCA_CASE_NUMBER', 'STATUS', 'LCA_CASE_EMPLOYER_STATE', 'LCA_CASE_EMPLOYER_NAME', 'TOTAL_WORKERS', 'LCA_CASE_WORKLOC1_STATE','LCA_CASE_JOB_TITLE', 'LCA_CASE_SOC_NAME', 'VISA_CLASS']].copy()
df_H1b_2010_ = df_H1b_2010[['DECISION_DATE', 'LCA_CASE_NUMBER', 'STATUS', 'LCA_CASE_EMPLOYER_STATE', 'LCA_CASE_EMPLOYER_NAME', 'TOTAL_WORKERS', 'WORK_LOCATION_STATE1','LCA_CASE_JOB_TITLE', 'LCA_CASE_SOC_NAME']].copy() 
df_H1b_2009_ = df_H1b_2009[['Decision_Date', 'LCA_CASE_NUMBER', 'STATUS', 'LCA_CASE_EMPLOYER_STATE', 'LCA_CASE_EMPLOYER_NAME', 'TOTAL_WORKERS', 'LCA_CASE_WORKLOC1_STATE','LCA_CASE_JOB_TITLE', 'LCA_CASE_SOC_NAME', 'VISA_CLASS']].copy()
df_H1b_2009_2_ = df_H1b_2009_2[['DOL_DECISION_DATE', 'CASE_NO', 'APPROVAL_STATUS', 'EMPLOYER_STATE', 'EMPLOYER_NAME', 'NBR_IMMIGRANTS', 'STATE_1','JOB_TITLE', 'OCCUPATIONAL_TITLE', 'PROGRAM_DESIGNATION']].copy()

#Renaming them
df_H2b_2015_.columns = ["DECISION_DATE", "CASE_NUMBER", "CASE_STATUS", "EMPLOYER_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", "PREVAILING_WAGE", "BASIC_RATE_OF_PAY", 'WORKSITE_STATE', 'JOB_TITLE', 'SOC_TITLE', "VISA_CLASS"]
df_H2b_2014_.columns = ["DECISION_DATE", "CASE_NUMBER", "CASE_STATUS", "EMPLOYER_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", "PREVAILING_WAGE", "BASIC_RATE_OF_PAY", 'WORKSITE_STATE', 'JOB_TITLE', 'SOC_TITLE', "VISA_CLASS"]
df_H2b_2013_.columns = ["DECISION_DATE", "CASE_NUMBER", "CASE_STATUS", "EMPLOYER_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", "PREVAILING_WAGE", "BASIC_RATE_OF_PAY", 'WORKSITE_STATE', 'JOB_TITLE', 'SOC_TITLE', "VISA_CLASS"]
df_H2b_2012_.columns = ["DECISION_DATE", "CASE_NUMBER", "CASE_STATUS", "EMPLOYER_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", "PREVAILING_WAGE", "BASIC_RATE_OF_PAY", 'WORKSITE_STATE', 'JOB_TITLE', 'SOC_TITLE', "VISA_CLASS"]
df_H2b_2011_.columns = ["DECISION_DATE", "CASE_NUMBER", "CASE_STATUS", "EMPLOYER_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", "PREVAILING_WAGE", "BASIC_RATE_OF_PAY", 'WORKSITE_STATE', 'JOB_TITLE', 'SOC_TITLE', "VISA_CLASS"]
df_H2b_2010_.columns = ["DECISION_DATE", "CASE_NUMBER", "CASE_STATUS", "EMPLOYER_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", "PREVAILING_WAGE", "BASIC_RATE_OF_PAY", 'WORKSITE_STATE', 'JOB_TITLE', 'SOC_TITLE', "VISA_CLASS"]
df_H2b_2009_.columns = ["DECISION_DATE", "CASE_NUMBER", "CASE_STATUS", "EMPLOYER_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", "PREVAILING_WAGE", "BASIC_RATE_OF_PAY", 'WORKSITE_STATE', 'JOB_TITLE',]

df_H2a_2015_.columns = ["DECISION_DATE", "CASE_NUMBER", "CASE_STATUS", "EMPLOYER_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", "BASIC_UNIT_OF_PAY", "BASIC_RATE_OF_PAY", 'WORKSITE_STATE', 'JOB_TITLE', 'SOC_TITLE',"VISA_CLASS"]
df_H2a_2014_.columns = ["DECISION_DATE", "CASE_NUMBER", "CASE_STATUS", "EMPLOYER_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", "BASIC_UNIT_OF_PAY", "BASIC_RATE_OF_PAY", 'WORKSITE_STATE', 'JOB_TITLE', 'SOC_TITLE',"VISA_CLASS"]
df_H2a_2013_.columns = ["DECISION_DATE", "CASE_NUMBER", "CASE_STATUS", "EMPLOYER_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", "BASIC_UNIT_OF_PAY", "BASIC_RATE_OF_PAY", 'WORKSITE_STATE', 'JOB_TITLE', "VISA_CLASS"]
df_H2a_2012_.columns = ["DECISION_DATE", "CASE_NUMBER", "CASE_STATUS", "EMPLOYER_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", "BASIC_UNIT_OF_PAY", "BASIC_RATE_OF_PAY", 'WORKSITE_STATE', 'JOB_TITLE', "VISA_CLASS"]
df_H2a_2011_.columns = ["DECISION_DATE", "CASE_NUMBER", "CASE_STATUS", "EMPLOYER_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", "BASIC_UNIT_OF_PAY", "BASIC_RATE_OF_PAY", 'WORKSITE_STATE', 'JOB_TITLE', "VISA_CLASS"]
df_H2a_2010_.columns = ["DECISION_DATE", "CASE_NUMBER", "CASE_STATUS", "EMPLOYER_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", "BASIC_UNIT_OF_PAY", "BASIC_RATE_OF_PAY", 'WORKSITE_STATE', 'JOB_TITLE']
df_H2a_2009_.columns = ["DECISION_DATE", "CASE_NUMBER", "CASE_STATUS", "EMPLOYER_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", "BASIC_UNIT_OF_PAY", "BASIC_RATE_OF_PAY", 'WORKSITE_STATE', 'JOB_TITLE']

df_H1b_2015_.columns = ["DECISION_DATE", "CASE_NUMBER", "CASE_STATUS", "EMPLOYER_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", 'WORKSITE_STATE', 'JOB_TITLE', 'SOC_TITLE',"VISA_CLASS"]
df_H1b_2014_.columns = ["DECISION_DATE", "CASE_NUMBER", "CASE_STATUS", "EMPLOYER_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", 'WORKSITE_STATE', 'JOB_TITLE', 'SOC_TITLE',"VISA_CLASS"]
df_H1b_2013_.columns = ["DECISION_DATE", "CASE_NUMBER", "CASE_STATUS", "EMPLOYER_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", 'WORKSITE_STATE', 'JOB_TITLE', 'SOC_TITLE',"VISA_CLASS"]
df_H1b_2012_.columns = ["DECISION_DATE", "CASE_NUMBER", "CASE_STATUS", "EMPLOYER_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", 'WORKSITE_STATE', 'JOB_TITLE', 'SOC_TITLE',"VISA_CLASS"]
df_H1b_2011_.columns = ["DECISION_DATE", "CASE_NUMBER", "CASE_STATUS", "EMPLOYER_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", 'WORKSITE_STATE', 'JOB_TITLE', 'SOC_TITLE',"VISA_CLASS"]
df_H1b_2010_.columns = ["DECISION_DATE", "CASE_NUMBER", "CASE_STATUS", "EMPLOYER_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", 'WORKSITE_STATE', 'JOB_TITLE','SOC_TITLE']
df_H1b_2009_.columns = ["DECISION_DATE", "CASE_NUMBER", "CASE_STATUS", "EMPLOYER_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", 'WORKSITE_STATE', 'JOB_TITLE','SOC_TITLE', "VISA_CLASS"]
df_H1b_2009_2_.columns = ["DECISION_DATE", "CASE_NUMBER", "CASE_STATUS", "EMPLOYER_STATE", "EMPLOYER_NAME", "NBR_WORKERS_CERTIFIED", 'WORKSITE_STATE', 'JOB_TITLE','SOC_TITLE', "VISA_CLASS"]

df = pd.concat([df_H1b_2009_2_, df_H1b_2009_, df_H1b_2010_, df_H1b_2011_, df_H1b_2012_, df_H1b_2013_, df_H1b_2014_, df_H1b_2015_, df_H2b_2015_, df_H2b_2014_, df_H2b_2013_, df_H2b_2012_, df_H2b_2011_, df_H2b_2010_, df_H2b_2009_, df_H2a_2015_, df_H2a_2014_, df_H2a_2013_, df_H2a_2012_, df_H2a_2011_, df_H2a_2010_, df_H2a_2009_], ignore_index=True)

def certification(x):
    if x == 'Certified - Full':
        return 'CERTIFIED'
    elif x == 'DETERMINATION ISSUED - CERTIFICATION':
        return 'CERTIFIED'
    elif x == 'Certification':
        return 'CERTIFIED'
    #This is okay, as when it's partial only a certain number of workers are then certified
    elif x == 'Certified - Partial':
        return 'CERTIFIED'
    elif x == 'Partial Certification':
        return 'CERTIFIED'
    elif x == 'DETERMINATION ISSUED - CERTIFICATION EXPIRED':
        return 'CERTIFIED'
    elif x == 'Partial Certified':
        return 'CERTIFIED'
    elif x == 'PARTIAL CERTIFIED':
        return 'CERTIFIED'
    elif x == 'PARTIAL CERTIFIED':
        return 'CERTIFIED'
    elif x == 'DETERMINATION ISSUED - PARTIAL CERTIFICATION':
        return 'CERTIFIED'
    elif x == 'Certified':
        return 'CERTIFIED'
    else:
        return x

df['CASE_STATUS'] = df['CASE_STATUS'].apply(certification)

df['CASE_STATUS'].value_counts()

df[df['CASE_STATUS'] == 'CERTIFIED']['NBR_WORKERS_CERTIFIED'].sum()

df.to_csv('Temporary_Worker_Visas_2009_2015.csv')

df = pd.read_csv('data/Temporary_Worker_Visas_2009_2015.csv', dtype='str')

df.info()

def parse_date(str_date):
    try:
        return dateutil.parser.parse(str_date)
    except:
        None

def certifications(str_):
    try:
        return int(str_)
    except:
        None

df['DECISION_DATE'] = df['DECISION_DATE'].apply(parse_date)

df['NBR_WORKERS_CERTIFIED'] = df['NBR_WORKERS_CERTIFIED'].apply(certifications)

df.index = df['DECISION_DATE']

df.info()

fig, ax = plt.subplots(figsize =(10,5), facecolor='White')
df.resample('M')['CASE_NUMBER'].count().plot(ax=ax)
ax.set_title("Temporary Working Visas to the United States Issued 2009 - 2015", fontname='DIN Condensed', fontsize=24)
plt.savefig('Temp_Visas.png', transparent=True, bbox_inches='tight')

fig, ax = plt.subplots(figsize =(10,5), facecolor='White')
df.resample('A')['CASE_NUMBER'].count().plot(ax=ax)
ax.set_title("Temporary Working Visas to the United States Issued 2009 - 2015", fontname='DIN Condensed', fontsize=24)
plt.savefig('Temp_Visas.png', transparent=True, bbox_inches='tight')

fig, ax = plt.subplots(figsize =(10,5), facecolor='White')
df.resample('Q')['CASE_NUMBER'].count().plot(ax=ax)
ax.set_title("Temporary Working Visas to the United States Issued 2009 - 2015", fontname='DIN Condensed', fontsize=24)
plt.savefig('Temp_Visas.png', transparent=True, bbox_inches='tight')

df['WORKSITE_STATE'].value_counts().head(10)

CERTIFIED_df = df[df['CASE_STATUS'] == 'CERTIFIED']

NBR_WORKERS_PERSTATE = pd.DataFrame(CERTIFIED_df.groupby('WORKSITE_STATE')['NBR_WORKERS_CERTIFIED'].sum().sort_values(ascending=False))

NBR_WORKERS_PERSTATE.reset_index(inplace=True)

def int_(x):
    return int(x)

NBR_WORKERS_PERSTATE['NBR_WORKERS_CERTIFIED'] = NBR_WORKERS_PERSTATE['NBR_WORKERS_CERTIFIED'].apply(int)

NBR_WORKERS_PERSTATE.head()

# Looking into which job is top per State

Worksite_state_list = df['WORKSITE_STATE'].tolist()

Unique_list = set(Worksite_state_list)

for x in Unique_list:
    Job = df[df['WORKSITE_STATE'] == x].groupby('SOC_TITLE')['NBR_WORKERS_CERTIFIED'].sum().sort_values(ascending=False).head(1)
    print(x, ':', Job)

# Work rate and people highered
Top_jobs_by_state = []
for x in Unique_list:
    Job = df[df['WORKSITE_STATE'] == x].groupby('SOC_TITLE')['NBR_WORKERS_CERTIFIED'].sum().sort_values(ascending=False).head(1)
    Job = str(Job)
    Job = Job.replace('SOC_TITLE', '')
    Job = Job.replace('/n', '')
    Job = Job.replace('Name: NBR_WORKERS_CERTIFIED, dtype: float64', '')
    
    State_jobs = {'state:': x,
                 'job': Job}
    
    Top_jobs_by_state.append(State_jobs)

df_top_jobs = pd.DataFrame.from_dict(Top_jobs_by_state)

df_top_jobs.head()

df_top_jobs.to_csv('Top_jobs_states.csv')

#Getting certified workers for 2015
CERTIFIED_2015 = CERTIFIED_df['2015']

VISA_Counts_per_State = pd.DataFrame(CERTIFIED_2015.groupby('WORKSITE_STATE')['NBR_WORKERS_CERTIFIED'].sum().sort_values(ascending=False))

VISA_Counts_per_State_2015 = VISA_Counts_per_State.reset_index()

VISA_Counts_per_State.head()

df_workforce = pd.read_csv('data/STATE_WORKING_FORCE_RATE.csv')

df_workforce.head()

#Bringing in number of foreign temporary Workers per state
df_mapping_data = VISA_Counts_per_State.merge(df_workforce, left_on = 'WORKSITE_STATE', right_on = 'State')

df_mapping_data = df_mapping_data.merge(df_top_jobs, left_on = 'WORKSITE_STATE', right_on = 'state:')

df_mapping_data['TEMP WORKERS PERCENTAGE'] = df_mapping_data['NBR_WORKERS_CERTIFIED'] / df_mapping_data['Working Force'] * 100

df_mapping_data.sort_values('TEMP WORKERS PERCENTAGE', ascending=False)

del df_mapping_data['State']

del df_mapping_data['state:']

df_mapping_data.to_csv('VISA_DATA_BY_STATE_to_be_mapped.csv')

df_mapping_data = pd.read_csv('data/VISA_DATA_BY_STATE_to_be_mapped.csv')

df_mapping_data['TEMP WORKERS PERCENTAGE'].hist(bins=[0, 0.25, .5, .8, 1.2, 1.7], histtype = 'stepfilled')
#r = 0, 0.25, .5, .8, 1.2, 1.7
#histfit(r)
#Using this to help create a waving line:
#http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.hist
plt.savefig('HIST_DISTRIBUTION.pdf', transparent=True, bbox_inches='tight')

df_mapping_data['TEMP WORKERS PERCENTAGE'].describe()

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

mu = 0.622489  # mean of distribution
sigma = 0.367315  # standard deviation of distribution
x = mu + sigma * df_mapping_data['TEMP WORKERS PERCENTAGE']
num_bins = 5

n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='green', alpha=0.5)
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--')
plt.subplots_adjust(left=0.15)
plt.show()







df.info()

df_H1B = df[df['VISA_CLASS'] == 'H-1B']

df_H1B['CASE_STATUS'].value_counts()

df_H1b_2010 = pd.read_excel('https://www.foreignlaborcert.doleta.gov/docs/lca/H-1B_FY2010.xlsx')

df_H1b_2010.info()



