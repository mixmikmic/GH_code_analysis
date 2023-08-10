get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

projects = pd.DataFrame.from_csv('opendata_projects.csv', index_col=None)

projects.tail()

area = len(projects['primary_focus_area'].drop_duplicates())
subject = len(projects['primary_focus_subject'].drop_duplicates())
area,subject

#1) Plot between Category and project success
p = projects[projects.funding_status != 'live'] # Considering only completed projects
p.loc[:,'date_completed'] = pd.to_datetime(p['date_completed'])
p.loc[:,'date_posted'] = pd.to_datetime(p['date_posted'])
p.loc[:,'date_expiration'] = pd.to_datetime(p['date_expiration'])
p1 = p.groupby(by='primary_focus_subject')
p2 = p[p.funding_status == 'completed']
p2.loc[:,'project_duration'] = p2['date_completed'] - p2['date_posted']
p2.loc[:,'project_expire_duration'] = p2['date_expiration'] - p2['date_posted']
p2.loc[:,'project_duration_days'] = p2['project_duration'].astype('timedelta64[D]')
p2.loc[:,'project_expire_duration_days'] = p2['project_expire_duration'].astype('timedelta64[D]')
p2 = p2[(p2.project_duration_days < p2.project_expire_duration_days) & (p2.project_duration_days < 200)]
p3 = p2.groupby(by='primary_focus_subject')
a = 100*p3.size()/p1.size()
b = p3['project_duration_days'].mean()

ax = a.plot(kind='bar',figsize=(20,5), ylim=(50,80))
ax.set_ylabel("Project Completion Percent")
ax.set_title('Plot between Subject category and Percentage of projects completed', fontsize=14)

bx = b.plot(kind='bar',figsize=(20,5), ylim=(10,60))
bx.set_ylabel("Project Duration in days")
bx.set_title('Plot between Subject category and Average project duration in days', fontsize=14)

# Low performing Subjects
a[a<a.mean()].sort_values()

# Top performing Subjects
a[a>a.mean()].sort_values(ascending=False)

#2) Scatter plot between Students Reached vs Duration of project
x = p2['students_reached']
y = list(p2['project_duration_days'])

fig, ax = plt.subplots(figsize=(20,10))
ax.scatter(x, y,alpha=0.2)

ax.set_xlim([0, 1000])
ax.set_xlabel('StudentsReached', fontsize=12)
ax.set_ylabel('Project Duration in days', fontsize=12)
ax.set_title('Scatter plot between Students Reached vs Duration of project', fontsize=14)

ax.grid(True)
fig.tight_layout()

# Line chart between Students Reached vs Duration of project
c = p2.groupby(p2.students_reached)
c = c['project_duration_days'].mean()
cx = c.plot(kind='line',figsize=(20,5), xlim=(0,1000), ylim=(0,160))
cx.set_ylabel("Project Duration in days")

# 3) Plot between poverty level and status
p = projects[projects.funding_status != 'live']
p1 = p.groupby(by='poverty_level')
print 'Number of poverty levels: ',len(p1)
p2 = p[p.funding_status == 'completed']
p2 = p2.groupby(by='poverty_level')
a = 100*p2.size()/p1.size()
b = p1['total_donations'].mean()

ax = a.plot(kind='bar',figsize=(10,5),ylim=(50,80))
ax.set_ylabel("Project Completion Percent")
ax.set_title('Plot between Poverty Level and Percentage of projects completed', fontsize=14)

# Let's do a Hypothesis test on the significance of difference of sucesses of Highest poverty schoold vs remaining schools
p1 = p[p.poverty_level == 'highest poverty']
p2 = p[p.poverty_level != 'highest poverty']
p1_p = float(len(p1[p1.funding_status == 'completed'])) / len(p1)
p2_p = float(len(p2[p2.funding_status == 'completed'])) / len(p2)
print'Sucess rates of both the groups:',p1_p,p2_p
# Assuming null hypothesis - h0 - No significance of poverty level "highest"
# Combined p
p3 = float(len(p1[p1.funding_status == 'completed']) + len(p2[p2.funding_status == 'completed'])) / (len(p1)+len(p2))
mean_diff = 0 #(Null hypothesis)
var_p1 = (p3*(1-p3))/len(p1)
var_p2 = (p3*(1-p3))/len(p2)
var_diff = var_p1 + var_p2
std_diff = np.sqrt(var_diff)
print 'Mean and Std of sampling dist p1-p2',mean_diff,std_diff

# Diff b/w sample means
mean_sample_diff = p1_p - p2_p

#z-score
z_score = (mean_diff-mean_sample_diff)/std_diff
print'z-score',z_score

# p-Value
p_value = stats.norm.cdf(z_score)
print 'P-Value =',p_value

# Plot between poverty level and donation amount
bx = b.plot(kind='bar',figsize=(10,5),ylim=(300,500))
bx.set_ylabel("Total Donation Amount")

# Finding the success of a school based on it's type
p1 = p[p.school_charter == 't']
p2 = p[p.school_magnet == 't']
p3 = p[p.school_year_round == 't']
p4 = p[p.school_nlns == 't']
p5 = p[p.school_kipp == 't']
p6 = p[p.school_charter_ready_promise == 't']
p7 = p[(p.school_charter_ready_promise == 'f') & (p.school_kipp == 'f') & (p.school_nlns == 'f') & (p.school_year_round == 'f') & (p.school_magnet == 'f') & (p.school_charter == 'f')]
p8 = p[(p.school_charter_ready_promise == 't') | (p.school_kipp == 't') | (p.school_nlns == 't') | (p.school_year_round == 't') | (p.school_magnet == 't') | (p.school_charter == 't')]
#p8=p[p.school_charter == 't'&((p.school_magnet == 't')|(p.school_year_round == 't')|(p.school_nlns == 't')|(p.school_kipp == 't')|(p.school_charter_ready_promise == 't'))]

p_p = float(len(p[p.funding_status == 'completed'])) / len(p)
p1_p = float(len(p1[p1.funding_status == 'completed'])) / len(p1)
p2_p = float(len(p2[p2.funding_status == 'completed'])) / len(p2)
p3_p = float(len(p3[p3.funding_status == 'completed'])) / len(p3)
p4_p = float(len(p4[p4.funding_status == 'completed'])) / len(p4)
p5_p = float(len(p5[p5.funding_status == 'completed'])) / len(p5)
p6_p = float(len(p6[p6.funding_status == 'completed'])) / len(p6)
p7_p = float(len(p7[p7.funding_status == 'completed'])) / len(p7)
p8_p = float(len(p8[p8.funding_status == 'completed'])) / len(p8)

print 'Proportions of each category:',p_p,p1_p,p2_p,p3_p,p4_p,p5_p,p6_p,p7_p,p8_p

# Hypothesis test 
print 'Sucess rates of two groups:',p7_p,p8_p
# Assuming null hypothesis - h0 - No significance of school type
# Combined p
p9 = float(len(p7[p7.funding_status == 'completed']) + len(p8[p8.funding_status == 'completed'])) / (len(p7)+len(p8))
mean_diff = 0 #(Null hypothesis)
var_p7 = (p9*(1-p9))/len(p7)
var_p8 = (p9*(1-p9))/len(p8)
var_diff = var_p7 + var_p8
std_diff = np.sqrt(var_diff)
print 'Mean and Std of sampling dist p8-p7',mean_diff,std_diff

# Diff b/w sample means
mean_sample_diff = p8_p - p7_p

#z-score
z_score = (mean_diff-mean_sample_diff)/std_diff
print'z-score',z_score

# p-Value
p_value = stats.norm.cdf(z_score)
print 'P-Value =',p_value

# Plot between state and the project completion proportion
p = p[p.school_state != 'La']
p1 = p.groupby(by='school_state')['_schoolid']
p2 = p[p.funding_status == 'completed']
p2 = p2.groupby(by='school_state')['_schoolid']
a = 100*p2.count()/p1.count()
a = a.sort_index()
ax = a.plot(kind='bar',figsize=(20,5),ylim=(50,83))
ax.set_ylabel("Project Completion Percent")
ax.set_title("Plot between project's state and the project completion proportion")

# Plot between Project Resource type and Donation Amount
p1 = projects.groupby(by='resource_type')
p1 = p1['total_price_including_optional_support'].mean()
p1.plot(kind='bar')

