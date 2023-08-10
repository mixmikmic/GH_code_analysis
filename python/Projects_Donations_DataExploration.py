get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math

donations = pd.DataFrame.from_csv('opendata_donations.csv', index_col=None).ix[:,0:23]
donations = donations.rename(columns=lambda x: x.strip()) # removing whitespaces from columns
donations = donations[(donations.donor_zip!='SC')&(donations.donor_zip!='NY')&(donations.donor_zip!='NJ')&(donations.donor_zip!='TX')]
donations.head(5)

projects = pd.DataFrame.from_csv('opendata_projects.csv', index_col=None)
projects.head(5)

# Join donations and projects data
projects_donations = projects.merge(donations, on='_projectid',how='inner')
projects_donations.head(5)

len(projects_donations)

#1) Proportions of state wise donations against state wise projects 
# Creating a n*n empty data frame for each state
states = projects_donations.school_state.drop_duplicates()
df = pd.DataFrame(columns = states)
for state in states:
    df.loc[state] = states

# Filling the data frame
p_g = projects_donations.groupby('school_state')
for state_p,group_p in p_g:
    #print state_p,'--',group_p.donation_total.sum()
    d_g = group_p.groupby('donor_state')
    for state_d,group_d in d_g:
        #print state_d, group_d.donation_total.sum()
        df.loc[state_p,state_d] = 100*(group_d.donation_total.sum())/(group_p.donation_total.sum())

df.fillna(0).head(5)

## Randomly picked few states - for data story sake
df.fillna(0)#.head(5)
df.loc[(df.index == 'NY')|(df.index == 'CA')|(df.index == 'DC')|(df.index == 'IA')|(df.index == 'NV')|(df.index == 'SC') ,(df.columns == 'NY')|(df.columns == 'CA')|(df.columns == 'DC')|(df.columns == 'IA')|(df.columns == 'NV')|(df.columns == 'SC')]

df[df.index == 'IA'].sum().sort_values(ascending=False).head(5)

# Identify donations that came from the teacher who created the project
# and assigning donor category
pds = projects_donations
pds.loc[:, 'donor_category'] = 'donor_other'
pds.loc[pds._teacher_acctid == pds._donor_acctid, 'donor_category'] = 'donor_teacher_project'
pds.loc[(pds._teacher_acctid != pds._donor_acctid) & (pds.is_teacher_acct == 't'), 'donor_category'] = 'donor_teacher_other'

pds_g = pds.groupby(by='donor_category')
pds_g = pds_g['donation_total'].mean()
pds_gx = pds_g.plot(kind='bar')
pds_gx.set_ylabel("Average Donation Amount")
pds_gx.set_title('Plot between donor category and Average donation amount', fontsize=14)

subjects = pds.primary_focus_subject.drop_duplicates()
df = pd.DataFrame(columns = ['Subjects','donor_teacher_project','donor_teacher_other','donor_other'])
df['Subjects'] = subjects
df = df.set_index('Subjects')

# Filling the data frame
pds1 = pds.groupby('primary_focus_subject')
for sub,group_s in pds1:
    pds2 = group_s.groupby('donor_category')
    for dc,group_dc in pds2:
        df.loc[sub,dc] = 100*(group_dc.donation_total.sum())/(group_s.donation_total.sum())
df.head(10)

a = pds[pds.donor_category == 'donor_teacher_project']
#a = a[(100 * a.donation_total / a.total_donations) > 33]
ax = a.groupby(by='primary_focus_subject')['donation_total'].mean().plot(kind='bar',figsize=(20,5))
ax.set_ylabel('Average Donation Amount')
ax.set_title('Plot between Subject and Donation amount for a teacher contributing to own project', fontsize=14)

b = pds[pds.donor_category == 'donor_teacher_other']
#a = a[(100 * a.donation_total / a.total_donations) > 33]
bx = b.groupby(by='primary_focus_subject')['donation_total'].mean().plot(kind='bar',figsize=(20,5))
bx.set_ylabel('Average Donation Amount')
bx.set_title('Plot between Subject and Donation amount for a teacher contributing to other project', fontsize=14)

c = pds[pds.donor_category == 'donor_other']
#a = a[(100 * a.donation_total / a.total_donations) > 33]
cx = c.groupby(by='primary_focus_subject')['donation_total'].mean().plot(kind='bar',figsize=(20,5))
cx.set_ylabel('Average Donation Amount')
cx.set_title('Plot between Subject and Donation amount for a non-teacher', fontsize=14)

p = pds[pds.funding_status != 'live'] # Considering only completed projects
p.loc[:,'date_completed'] = pd.to_datetime(p['date_completed'])
p.loc[:,'date_posted'] = pd.to_datetime(p['date_posted'])
p.loc[:,'date_expiration'] = pd.to_datetime(p['date_expiration'])
p1 = p.groupby(by='donor_category')
p2 = p[p.funding_status == 'completed']
p2.loc[:,'project_duration'] = p2['date_completed'] - p2['date_posted']
p2.loc[:,'project_expire_duration'] = p2['date_expiration'] - p2['date_posted']
p2.loc[:,'project_duration_days'] = p2['project_duration'].astype('timedelta64[D]')
p2.loc[:,'project_expire_duration_days'] = p2['project_expire_duration'].astype('timedelta64[D]')
p2 = p2[(p2.project_duration_days < p2.project_expire_duration_days) & (p2.project_duration_days < 200)]
p3 = p2.groupby(by='donor_category')
a = 100*p3.size()/p1.size()
b = p3['project_duration_days'].mean()

#a.plot(kind='bar')
ax = a.plot(kind='bar',figsize=(7,5),ylim=(50,100))
ax.set_ylabel("Project Completion Percent")

bx = b.plot(kind='bar',figsize=(7,5))
bx.set_ylabel("Project Duration")

