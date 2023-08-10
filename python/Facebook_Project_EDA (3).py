#Import the necessary packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
get_ipython().magic('matplotlib inline')

#Read the input file in variable facebook

path = 'C:/Manoj/Data Science/Foundation Projects/Facebook/pseudo_facebook.tsv'
facebook = pd.read_csv(path,sep='\t',parse_dates=[[3,4,2]])
facebook.tail()

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

#Lets see the number of Males in the facebook data

males = (facebook['gender']=='male').sum()
males

#No of Females in the facebook data

females = (facebook['gender']=='female').sum()
females

#Lets find out the No. of like received by Females

likes = facebook.groupby('gender')[['likes_received']].sum()
flikes = likes.ix['female']

#And No. of like received by Males
mlikes = likes.ix['male']
print(mlikes)
print(flikes)

#Lets draw Pie chart for population proportion of males and females

percent_pop = [males,females]

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

#Let's plot the scatter plot to see how no of likes is influenced by friend_count

plt.scatter(facebook['friend_count'],facebook['likes_received'])
plt.xlabel('friend_count')
plt.ylabel('likes_received')
plt.title('scatter plot for Likes_received vs Friends_count')

#Analysis - It seems there is no relation as such in specific between likes_received w.r.t friend_count.
# So, a general assumption of higher the friends higher will be likes doesnt seems to be correct at least by this plot.
# Although there are some specific outliers

#Now Let's see how the likes_received are getting influenced by age of a person

plt.scatter(facebook['age'],facebook['likes_received'])
plt.xlabel('Age')
plt.ylabel('likes_received')
plt.title('scatter plot for Likes_received vs Age')

#Analysis - It looks like the distribution is little rightly skewed from 0-25 years of age and again no of likes are 
# increasing near 40 or so. Outliers in the plot are also present but their cocentration is more near 20 years of the age of
# user.

# Lets see whether user can increase his/her 'likes_received' if he starts liking other users status/pic more often.

plt.scatter(facebook['likes'],facebook['likes_received'])
plt.xlabel('likes')
plt.ylabel('likes_received')
plt.title('scatter plot for Likes_received vs likes_made_by_user')

#Analysis :- Seems to be similar with other prevous plots and looks like there is no sepcial increase in likes_received
# even if user is liking other user profile/picture...wonder if its correct, I always thought the other way round...:)
# Although some outliers justifies that for having a high number of likes received one doesn't have to like more
# of others stuff.. :)

#Now last in this category, lets see how likes_received vary with the amount of tenure a user sepnts in facebook

plt.scatter(facebook['tenure'],facebook['likes_received'])
plt.xlabel('Tenure in days')
plt.ylabel('likes_received')
plt.title('scatter plot for Likes_received vs Tenure')

#Analysis :- The plot below is similar to little rightly skewed and it seems likes_received
#increases till first 300-500 days and then it starts decreasing. Outliers in this plot too are concentrated
# near the right skew peak. This proves that general assumption of having more number of likes is more likely to 
# happen if the user's tenure is high in facebook is not true. Because one can see even if user has spent max. tenure
# say more than 1500 days, his likes are lesses than the one who has only spent 500 days in facebook.

#Lets plot to see if there is any relation between friend_count & Age of a user.

plt.scatter(facebook['age'],facebook['friend_count'])
plt.xlabel('age')
plt.ylabel('friend_count')
plt.title('scatter plot for friend_count vs age')

#Analysis :- Again as rest of the plots, the no of friend_count vs age plot looks rightly skewed, chances of having more 
# friends if he is in age grup till 20-25 is high than as compared to later age groups. Although for every age there
# are some very high number of outliers.And one interesting fact, at intervals around 100 years or 110 years of age
# frind_count is gain at its peak.

#Lets plot to see how the plot is for friend_count & friendship_initiated for users.

plt.scatter(facebook['friendships_initiated'],facebook['friend_count'])
plt.xlabel('friendships_initiated')
plt.ylabel('friend_count')
plt.title('plot for friend_count vs friendships_intiated')

#Analysis :- This plot looks different, to some extent it seems friend_count has been directly proportional(atleast to 
# some extent) to no of riend_requests sent, as more the friendships one has initiated, his/her friend count 
# has always increased.

#Lets see one last to check how fiend_count vary w.r.t tenure of a user

plt.scatter(facebook['friend_count'],facebook['tenure'])
plt.xlabel('tenure in days')
plt.ylabel('friend_count')
plt.title('plot for friend_count vs tenure')

#Analysis :- This plot again seems to me a blurred example of right skewed distribution, it seems till the tenure of 1000
#days, friend_count is increasing & is at peaks.And for all the user whose tenure is incresing there is surge in friends count...

# Lets draw the correlation matrix to see the correlation impact of each attribute with each other.

corr=facebook.corr(method='spearman')
get_ipython().magic('matplotlib inline')
plt.title('Facebook Correlation Matrix')
sns.heatmap(corr)

#Analysis :- So, from the below we can make out that there is a positive co=relation between Friend_count & friend_ships
# initiated & similary we have positive correlation for various likes received/given with total number of likes.
# Another interesting aspect is friend_count/friendships_initiated is in near to negative correlation with age, this
# is what we have also seen in the above plots as well...(Like wise we csn do other investigations as well)

# Max number of likes received & friends_count for a particular age (males and females)

max_likes = facebook.groupby(['age','gender'])[['likes_received','friend_count']].max()
max_likes.head()

# Will be very interesting to create gender specific graphs against the same index, to see how for particular gender
# plot is getting influenced by particualar factors.

#Please suggest in case i am wrong or to add any!!

# The above plots are very generic and simple in nature but for next stage, visualization has to be done for more 
# complex/complicated use cases, so that some hidden/interesting facts can be found out...

# Data Transformation is required as there will be numerous age values and need to put in respective group to help 
# analyze Age related factors better

def age_compute(row):
    if row['age'] < 20:
        return '13-19'
    elif row['age'] >= 20 and row['age'] < 40:
        return '20-39'
    elif row['age'] >= 40 and row['age'] < 60:
        return '40-60'
    elif row['age'] >= 60:
        return '60 Plus'

facebook['age_group'] = facebook.apply(age_compute,axis=1)

facebook.tail()

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

# It is found that overall population is using more and more mobile than web for accessing facebook
# Now lets see how it differs from age point of view 

# Now lets see which age group is using mobile more than others

facebook_age_group = facebook.groupby('age_group').sum()
facebook_age_group

facebook_age_group.plot(facebook_age_group.index, 'mobile_likes', kind='bar', color='r')

# As expected young population of 20-39 range are using facebook through mobile more than others

