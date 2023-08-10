#Import the necessary packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
get_ipython().magic('matplotlib inline')

#Read the input file in variable facebook

path = 'G:/Personal/UpX/Foundation/DS_Foundation_Project/datasets/pseudo_facebook.tsv'
#facebook = pd.read_csv(path,sep='\t',parse_dates=[[3,4,2]])
facebook = pd.read_csv(path,delimiter='\t',parse_dates=[[3,4,2]])
facebook.tail()


#df=pd.read_csv('pseudo_facebook.tsv',delimiter='\t')
#df.head(5)

facebook['tenure_yrs']=facebook['tenure']/365
facebook['tenure_mnth']=facebook['tenure']/30
facebook['tenure_mnth']=facebook['tenure_mnth'].round(1)
facebook['tenure_yrs']=facebook['tenure_yrs'].apply(np.ceil)
facebook.head(5)

#facebook variable no of rows and columns
facebook.shape

#facebook variable column variables and data types
facebook.info()

#Renaming the date of birth coumn to shorter name as DOB and further setting index of te dataframe as userid

facebook.rename(columns = {'dob_year_dob_month_dob_day':'DOB'},inplace=True)
facebook = facebook.set_index('userid')
facebook.head()

#Use describe function to see the various aspects of facebbok data

facebook.describe()

#Likes received by tenure_yrs
likes_fb1 = facebook.groupby(['tenure_yrs'])[['likes_received']].sum()
likes_fb1['L_R-sum'] = facebook.groupby(['tenure_yrs'])[['likes_received']].sum()
likes_fb1['L_R-count'] = facebook.groupby(['tenure_yrs'])[['likes_received']].count()
likes_fb1['L_R-Average'] = facebook.groupby(['tenure_yrs'])[['likes_received']].mean().round(1)
#likes_fb1.drop(likes_fb1.columns[[0,2]],axis=1, inplace=True)
likes_fb1

get_ipython().magic('matplotlib inline')
#plt1 = likes_fb1[['L_R-sum','L_R-Average']].plot() # can't be done on same plot because of scale issue
plt1 = likes_fb1['L_R-sum'].plot()
# changes the size of the graph
fig = plt1.get_figure()
fig.set_size_inches(13.5, 9)
#plt1.figure(figsize=(20,10))
plt.xlabel('tanure_yrs')
plt.ylabel('likes_received')
plt.title('Total Likes Received')

plt1 = likes_fb1['L_R-Average'].plot()
# changes the size of the graph
fig = plt1.get_figure()
fig.set_size_inches(13.5, 9)
#plt1.figure(figsize=(20,10))
plt.xlabel('tanure_yrs')
plt.ylabel('likes_received')
plt.title('Average Likes Received')

plt1 = likes_fb1['L_R-count'].plot()
# changes the size of the graph
fig = plt1.get_figure()
fig.set_size_inches(13.5, 9)
#plt1.figure(figsize=(20,10))
plt.xlabel('tanure_yrs')
plt.ylabel('likes_received')
plt.title('Total Likes Received')

# plot likes_received by tenure_mnth
plt1 = likes_fb1.plot(likes_fb1.index,'likes_received', kind='line', color='g')

# changes the size of the graph
fig = plt1.get_figure()
fig.set_size_inches(13.5, 9)
#plt1.figure(figsize=(20,10))

# likes_received by tenure_mnth for female
gender_f=facebook[facebook['gender']=='female']
f_likes_fb1=gender_f.groupby(['tenure_yrs'])[['likes_received']].sum()
f_likes_fb1

# plot likes_received by tenure_mnth for female
plt_f = f_likes_fb1.plot(f_likes_fb1.index,'likes_received', kind='line', color='r')
# changes the size of the graph
fig = plt_f.get_figure()
fig.set_size_inches(13.5, 9)

#f_likes_fb1.plot.xlabel('tanure_yrs')
#plt_f.ylabel('likes_received')
#plt_f.title('Total Likes Received (Female)')

# likes_received by tenure_mnth for male
gender_m=facebook[facebook['gender']=='male']
m_likes_fb1=gender_m.groupby(['tenure_yrs'])[['likes_received']].sum()
m_likes_fb1

# Plot likes_received by tenure_mnth for female
plt_m = m_likes_fb1.plot(m_likes_fb1.index,'likes_received', kind='line', color='b')
# changes the size of the graph
fig = plt_m.get_figure()
fig.set_size_inches(13.5, 9)

likes_fb1.describe()

f_likes_fb1.describe()

m_likes_fb1.describe()

# merge female and male trends
# unable to do needs some help *****

f_likes_fb1=f_likes_fb1[['tenure_mnth','likes_received']]
m_likes_fb1=m_likes_fb1[['tenure_mnth','likes_received']]
gender_likes=pd.merge(f_likes_fb1, m_likes_fb1, on='tenure_mnth')
gender_likes.columns=['tenure_mnth','female','male']
#gender_likes[['female','male']].plot()

#fb_fr_counts = pd.merge(fb_males_fr_count,fb_females_fr_count,on='tenure_yrs')
#gender_likes.columns=['tenure','males','females']
plt.plot(gender_likes[['males','females']],linewidth=2.0)
plt.xlabel('tenure (in months)',fontsize = 'x-large')
plt.ylabel('No. of new friends made per year',fontsize = 'x-large')
plt.legend(['males','females'],loc='upper right',fontsize = 'x-large')
plt.show()

# codes by Aik Lee
likes_fb=facebook[['tenure_yrs','likes_received']]
likes_fb=likes_fb.groupby('tenure_yrs').mean().unstack().reset_index()
likes_fb['avglikesyr']=likes_fb[0]/likes_fb['tenure_yrs']
likes_fb

# codes by Aik Lee
likes_fb=likes_fb[['tenure_yrs','avglikesyr']]
likes_fb

# codes by Aik Lee
get_ipython().magic('matplotlib inline')
likes_fb['avglikesyr'].plot()

# codes by Aik Lee
female_likes_fb=facebook[facebook['gender']=='female']
female_likes_fb=female_likes_fb[['tenure_yrs','likes_received']]
female_likes_fb=female_likes_fb.groupby('tenure_yrs').mean().unstack().reset_index()
female_likes_fb['avglikesyr']=female_likes_fb[0]/female_likes_fb['tenure_yrs']
female_likes_fb=female_likes_fb[['tenure_yrs','avglikesyr']]
female_likes_fb

# codes by Aik Lee
female_likes_fb['avglikesyr'].plot()

# codes by Aik Lee
male_likes_fb=facebook[facebook['gender']=='male']
male_likes_fb=male_likes_fb[['tenure_yrs','likes_received']]
male_likes_fb=male_likes_fb.groupby('tenure_yrs').mean().unstack().reset_index()
male_likes_fb['avglikesyr']=male_likes_fb[0]/male_likes_fb['tenure_yrs']
male_likes_fb=male_likes_fb[['tenure_yrs','avglikesyr']]
male_likes_fb

# codes by Aik Lee
male_likes_fb['avglikesyr'].plot()

# codes by Aik Lee
gender_likes=pd.merge(female_likes_fb, male_likes_fb, on='tenure_yrs')
gender_likes.columns=['tenure_yrs','female','male']
gender_likes[['female','male']].plot()

#-Diego
likes_Vs_age=plt.scatter(facebook['likes'],facebook['age'])
print('The mode of age   for facebook users is :',(facebook['age']).mode())
print('The mean of age  for the facebbok users is',(facebook['age']).mean())
print('The max of age for the facebook users is ', (facebook['age']).max())
print('The minimum age fro the facebook user is ', (facebook['age']).min())
                         

# Diego code
likes_Vs_age=plt.scatter(facebook['age'],facebook['likes'])

# changes the size of the graph
fig = likes_Vs_age.get_figure()
fig.set_size_inches(13.5, 9)

plt.xlabel('age')
plt.ylabel('likes')
plt.title('scatter plot for age vs likes')

# replaced outliers with mean to visualize the relationship between age and likes

#df=pd.DataFrame({'a':[facebook['age']],'b':[facebook['likes']]})
#print(df)

def replace(data):
    mean, std = data.mean(), data.std()
    outliers = (data - mean).abs() > 3*std
    data[outliers] = mean        # or "group[~outliers].mean()"
    return data

filtered_L_d = replace(facebook['likes'])
print filtered_L_d

#df.groupby('a').transform(replace)

# plot with outliers smoothed
likes_Vs_age_outliersrm=plt.scatter(facebook['age'],filtered_L_d)

# changes the size of the graph
fig = likes_Vs_age_outliersrm.get_figure()
fig.set_size_inches(13.5, 9)

plt.xlabel('age')
plt.ylabel('likes')
plt.title('scatter plot for age vs likes (outliers replaced with mean)')

likes_received_Vs_age=plt.scatter(facebook['age'],facebook['likes_received'])

# changes the size of the graph
fig = likes_received_Vs_age.get_figure()
fig.set_size_inches(13.5, 9)

plt.xlabel('age')
plt.ylabel('likes_received')
plt.title('scatter plot for age vs likes_received')

# replaced outliers with mean to visualize the relationship between age and likes_received

#df=pd.DataFrame({'a':[facebook['age']],'b':[facebook['likes']]})
#print(df)

def replace(data):
    mean, std = data.mean(), data.std()
    outliers = (data - mean).abs() > 3*std
    data[outliers] = mean        # or "group[~outliers].mean()"
    return data

filtered_LR_d = replace(facebook['likes_received'])
print filtered_LR_d

#df.groupby('a').transform(replace)

likes_received_Vs_age_outliersrm=plt.scatter(facebook['age'],filtered_LR_d)

# changes the size of the graph
fig = likes_received_Vs_age_outliersrm.get_figure()

fig.set_size_inches(13.5, 9)

plt.xlabel('age')
plt.ylabel('likes_received')
plt.title('scatter plot for age vs likes_received (outliars replaced by mean)')

likes_received_Vs_likes=plt.scatter(facebook['likes'],facebook['likes_received'])

# changes the size of the graph
fig = likes_received_Vs_likes.get_figure()
fig.set_size_inches(13.5, 9)

plt.xlabel('likes')
plt.ylabel('likes_received')
plt.title('scatter plot for likes vs likes_received')

# Diego
bp_age_gender = facebook.boxplot(column='age', by='gender')

# changes the size of the graph
fig = bp_age_gender.get_figure()

fig.set_size_inches(13.5, 9)

#Diego
bp_fc_gender = facebook.boxplot(column='friend_count', by='gender')
plt.ylim((-10,600))

# changes the size of the graph
fig = bp_fc_gender.get_figure()

fig.set_size_inches(13.5, 9)

#Diego
bp_tenure_gender = facebook.boxplot(column='tenure', by='gender')
plt.ylim((-100,2000))


# changes the size of the graph
fig = bp_tenure_gender.get_figure()

fig.set_size_inches(13.5, 9)

m = (facebook['gender']=='male').sum()
m

f = (facebook['gender']=='female').sum()
f

# Deigo
#Lets draw Pie chart for population proportion of males and females

percent_pop = [m,f]

plt.pie(
      percent_pop,
      labels = ['Males','Females'],
      shadow = False,
       colors = ['blue','pink'],
       explode = (0.15 , 0),
       startangle =30,
       autopct = '%1.1f%%'
)
plt.axis('equal')
plt.title("Gender Proportion")
plt.tight_layout()
plt.show()

# Deigo
#Lets find out the No. of like received by Females

likes = facebook.groupby('gender')[['likes_received']].sum()
flikes = likes.ix['female']

#And No. of like received by Males
mlikes = likes.ix['male']
print(mlikes)
print(flikes)

#Deigo
#As we saw the population proportion of males vs feamles, lets see the proportion of facebook likes for males vs females

percent_likes = [mlikes,flikes]

plt.pie(
      percent_likes,
      labels = ['Male_Likes_received','Female_Likes_received'],
      shadow = False,
       colors = ['blue','pink'],
       explode = (0.15 , 0),
       startangle = 90,
       autopct = '%1.1f%%'
)
plt.axis('equal')
plt.title("Likes_Received Proportion")
plt.tight_layout()
plt.show()

#Analysis :- So, we saw in spite of having the more percentage population proportion of males , the likes proportion of females is
#much higher than males..:) 

# Diego
#Lets plot to see how the plot is for friend_count & friendship_initiated for users.

fr_initiated_vs_fr = plt.scatter(facebook['friendships_initiated'],facebook['friend_count'])

# changes the size of the graph
fig = fr_initiated_vs_fr.get_figure()
fig.set_size_inches(13.5, 9)

plt.xlabel('friendships_initiated')
plt.ylabel('friend_count')
plt.title('plot for friend_count vs friendships_intiated')

gender_f=facebook[facebook['gender']=='female']

fr_initiated_vs_fr = plt.scatter(gender_f['friendships_initiated'],gender_f['friend_count'], color='r')

# changes the size of the graph
fig = fr_initiated_vs_fr.get_figure()
fig.set_size_inches(13.5, 9)

plt.xlabel('friendships_initiated')
plt.ylabel('friend_count')
plt.title('plot for friend_count vs friendships_intiated (Female)')

gender_m=facebook[facebook['gender']=='male']

fr_initiated_vs_fr = plt.scatter(gender_m['friendships_initiated'],gender_m['friend_count'], color='b')

# changes the size of the graph
fig = fr_initiated_vs_fr.get_figure()
fig.set_size_inches(13.5, 9)

plt.xlabel('friendships_initiated')
plt.ylabel('friend_count')
plt.title('plot for friend_count vs friendships_intiated (Male)')

# Diego
#Lets see one last to check how fiend_count vary w.r.t tenure of a user

fr_cnt_vs_tenure_days = plt.scatter(facebook['friend_count'],facebook['tenure'])
plt.xlabel('tenure in days')
plt.ylabel('friend_count')
plt.title('plot for friend_count vs tenure in days')

# changes the size of the graph
fig = fr_cnt_vs_tenure_days.get_figure()
fig.set_size_inches(13.5, 9)

#Analysis :- This plot again seems to me a blurred example of right skewed distribution, it seems till the tenure of 1000
#days, friend_count is increasing & is at peaks.And for all the user whose tenure is incresing there is surge in friends count...

#Diego
# Lets draw the correlation matrix to see the correlation impact of each attribute with each other.

corr=facebook.corr(method='spearman')
get_ipython().magic('matplotlib inline')
plt.title('Facebook Correlation Matrix')
corr_hm = sns.heatmap(corr)

# changes the size of the graph
fig = corr_hm.get_figure()
fig.set_size_inches(13.5, 9)


#Analysis :- So, from the below we can make out that there is a positive co=relation between Friend_count & friend_ships
# initiated & similary we have positive correlation for various likes received/given with total number of likes.
# Another interesting aspect is friend_count/friendships_initiated is in near to negative correlation with age, this
# is what we have also seen in the above plots as well...(Like wise we csn do other investigations as well)

#Determine how many people are using Mobile over Web for accessing facebook

mobile_usage = (facebook['mobile_likes']).sum()
web_usage = (facebook['www_likes']).sum()
mobile_usage
web_usage

percent_pop = [mobile_usage,web_usage]

plt.pie(
      percent_pop,
      labels = ['Mobile Users','Web Users'],
      shadow = False,
       #colors = ['blue','pink'],
       explode = (0.15 , 0),
       startangle = 90,
       autopct = '%1.1f%%'
)
plt.axis('equal')
plt.title("Device Proportion")
plt.tight_layout()
plt.show()

#Diego
# Data Transformation is required as there will be numerous age values and need to put in respective group to help 
# analyze Age related factors better

def age_compute(row):
    if row['age'] < 20:
        return '13-19'
    elif row['age'] >= 20 and row['age'] < 30:
        return '20-29'
    elif row['age'] >= 30 and row['age'] < 40:
        return '30-39'
    elif row['age'] >= 40 and row['age'] < 50:
        return '40-49'
    elif row['age'] >= 50 and row['age'] < 60:
        return '50-59'
    elif row['age'] >= 60:
        return '60 Plus'

facebook['age_group'] = facebook.apply(age_compute,axis=1)

facebook.tail()

likes_fb1_agegrp=facebook.groupby(['age_group'])[['mobile_likes','www_likes']].sum()
likes_fb1_agegrp

#Lets plot bar graph for 'Age Group Vs Likes'

n_groups = 6
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = .85
 
rects1 = plt.bar(index, likes_fb1_agegrp['mobile_likes'], bar_width,
                 alpha=opacity,
                 color='blue',
                 label='Mobile')
 
rects2 = plt.bar(index + bar_width, likes_fb1_agegrp['www_likes'], bar_width,
                 alpha=opacity,
                 color='pink',
                 label='Web')
 
plt.xlabel('Age Groups')
plt.ylabel('Likes')
plt.title('Age Group Vs Likes')
mob_vs_web_by_agegrp = plt.xticks(index + bar_width, ('13-19', '20-29', '30-39', '40-49', '50-59', '60+'))
plt.legend()
 
plt.tight_layout()
plt.show()

# changes the size of the graph
#fig = mob_vs_web_by_agegrp.get_figure()
#fig.set_size_inches(13.5, 9)


gender_f=facebook[facebook['gender']=='female']
f_likes_fb1_agegrp=gender_f.groupby(['age_group'])[['mobile_likes','www_likes']].sum()
f_likes_fb1_agegrp

#Lets plot bar graph for 'Age Group Vs Likes (Female)'

n_groups = 6
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = .85
 
rects1 = plt.bar(index, f_likes_fb1_agegrp['mobile_likes'], bar_width,
                 alpha=opacity,
                 color='blue',
                 label='Mobile')
 
rects2 = plt.bar(index + bar_width, f_likes_fb1_agegrp['www_likes'], bar_width,
                 alpha=opacity,
                 color='pink',
                 label='Web')
 
plt.xlabel('Age Groups')
plt.ylabel('Likes (Total)')
plt.title('Age Group Vs Likes (Female)')
mob_vs_web_by_agegrp = plt.xticks(index + bar_width, ('13-19', '20-29', '30-39', '40-49', '50-59', '60+'))
plt.legend()
 
plt.tight_layout()
plt.show()

# changes the size of the graph
#fig = mob_vs_web_by_agegrp.get_figure()
#fig.set_size_inches(13.5, 9)

gender_m=facebook[facebook['gender']=='male']
m_likes_fb1_agegrp=gender_m.groupby(['age_group'])[['mobile_likes','www_likes']].sum()
m_likes_fb1_agegrp

#Lets plot bar graph for 'Age Group Vs Likes (Males)'

n_groups = 6
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = .85
 
rects1 = plt.bar(index, m_likes_fb1_agegrp['mobile_likes'], bar_width,
                 alpha=opacity,
                 color='blue',
                 label='Mobile')
 
rects2 = plt.bar(index + bar_width, m_likes_fb1_agegrp['www_likes'], bar_width,
                 alpha=opacity,
                 color='pink',
                 label='Web')
 
plt.xlabel('Age Groups')
plt.ylabel('Likes (Total)')
plt.title('Age Group Vs Likes (Males)')
mob_vs_web_by_agegrp = plt.xticks(index + bar_width, ('13-19', '20-29', '30-39', '40-49', '50-59', '60+'))
plt.legend()
 
plt.tight_layout()
plt.show()

# changes the size of the graph
#fig = mob_vs_web_by_agegrp.get_figure()
#fig.set_size_inches(13.5, 9)

likes_fb1_agegrpm=facebook.groupby(['age_group'])[['mobile_likes','www_likes']].mean()
likes_fb1_agegrpm

#Lets plot bar graph for 'Age Group Vs Likes'

n_groups = 6
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = .85
 
rects1 = plt.bar(index, likes_fb1_agegrpm['mobile_likes'], bar_width,
                 alpha=opacity,
                 color='blue',
                 label='Mobile')
 
rects2 = plt.bar(index + bar_width, likes_fb1_agegrpm['www_likes'], bar_width,
                 alpha=opacity,
                 color='pink',
                 label='Web')
 
plt.xlabel('Age Groups')
plt.ylabel('Likes (Average)')
plt.title('Age Group Vs Likes')
mob_vs_web_by_agegrp = plt.xticks(index + bar_width, ('13-19', '20-29', '30-39', '40-49', '50-59', '60+'))
plt.legend()
 
plt.tight_layout()
plt.show()

# changes the size of the graph
#fig = mob_vs_web_by_agegrp.get_figure()
#fig.set_size_inches(13.5, 9)

gender_m=facebook[facebook['gender']=='male']
m_likes_fb1_agegrpm=gender_m.groupby(['age_group'])[['mobile_likes','www_likes']].mean()
m_likes_fb1_agegrpm

#Lets plot bar graph for 'Age Group Vs Likes'

n_groups = 6
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = .85
 
rects1 = plt.bar(index, m_likes_fb1_agegrpm['mobile_likes'], bar_width,
                 alpha=opacity,
                 color='blue',
                 label='Mobile')
 
rects2 = plt.bar(index + bar_width, m_likes_fb1_agegrpm['www_likes'], bar_width,
                 alpha=opacity,
                 color='pink',
                 label='Web')
 
plt.xlabel('Age Groups')
plt.ylabel('Likes (Average)')
plt.title('Age Group Vs Likes (male)')
mob_vs_web_by_agegrp = plt.xticks(index + bar_width, ('13-19', '20-29', '30-39', '40-49', '50-59', '60+'))
plt.legend()
 
plt.tight_layout()
plt.show()

# changes the size of the graph
#fig = mob_vs_web_by_agegrp.get_figure()
#fig.set_size_inches(13.5, 9)

gender_m=facebook[facebook['gender']=='female']
f_likes_fb1_agegrpm=gender_m.groupby(['age_group'])[['mobile_likes','www_likes']].mean()
f_likes_fb1_agegrpm

#Lets plot bar graph for 'Age Group Vs Likes'

n_groups = 6
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = .85
 
rects1 = plt.bar(index, f_likes_fb1_agegrpm['mobile_likes'], bar_width,
                 alpha=opacity,
                 color='blue',
                 label='Mobile')
 
rects2 = plt.bar(index + bar_width, f_likes_fb1_agegrpm['www_likes'], bar_width,
                 alpha=opacity,
                 color='pink',
                 label='Web')
 
plt.xlabel('Age Groups')
plt.ylabel('Likes (Average)')
plt.title('Age Group Vs Likes (Female)')
mob_vs_web_by_agegrp = plt.xticks(index + bar_width, ('13-19', '20-29', '30-39', '40-49', '50-59', '60+'))
plt.legend()
 
plt.tight_layout()
plt.show()

# changes the size of the graph
#fig = mob_vs_web_by_agegrp.get_figure()
#fig.set_size_inches(13.5, 9)

likes_fb1_agegrp=facebook.groupby(['age_group'])[['mobile_likes','www_likes']].count()
likes_fb1_agegrp

likes_fb1_gender=facebook.groupby(['gender'])[['mobile_likes','www_likes']].sum()
likes_fb1_gender

#Lets plot bar graph for 'Age Group Vs Likes'

n_groups = 2
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = .85
 
rects1 = plt.bar(index, likes_fb1_gender['mobile_likes'], bar_width,
                 alpha=opacity,
                 color='blue',
                 label='Mobile')
 
rects2 = plt.bar(index + bar_width, likes_fb1_gender['www_likes'], bar_width,
                 alpha=opacity,
                 color='pink',
                 label='Web')
 
plt.xlabel('Gender')
plt.ylabel('Likes')
plt.title('Gender by Likes')
mob_vs_web_by_agegrp = plt.xticks(index + bar_width, ('Female', 'Male'))
plt.legend()
 
plt.tight_layout()
plt.show()

# changes the size of the graph
#fig = mob_vs_web_by_agegrp.get_figure()
#fig.set_size_inches(13.5, 9)

likes_fb1_ten_yrs=facebook.groupby(['tenure_yrs'])[['mobile_likes','www_likes']].sum()
likes_fb1_ten_yrs

# plot likes_received by tenure_mnth
plt_Tn_yrs_mob_web = likes_fb1_ten_yrs.plot(likes_fb1.index,['mobile_likes','www_likes'], kind='line', color=['g','b'])

# changes the size of the graph
fig = plt_Tn_yrs_mob_web.get_figure()
fig.set_size_inches(13.5, 9)
#plt1.figure(figsize=(20,10))



