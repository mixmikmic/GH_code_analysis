get_ipython().magic('matplotlib inline')

#import the standard packages and read the csv table into a dataframe
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
meetup_df  = pd.read_csv("meetups.csv")

type(meetup_df)

meetup_df.shape # 2259 rows and 9 columns

meetup_df.columns

meetup_df.head()

#Everthing was stored as a string during the scraping process. Therefore, we will need to
#convert some the of the columns to intergers before doing any analysis. But remove commas 
meetup_df['group_members'] = meetup_df['group_members'].str.replace(",","")
meetup_df['upcoming_meetings'] = meetup_df['upcoming_meetings'].str.replace(",","")
meetup_df['past_meetings'] = meetup_df['past_meetings'].str.replace(",","")
meetup_df['group_reviews'] = meetup_df['group_reviews'].str.replace(",","")

#Ask about why I have to run this twice to get floats as the dtype
meetup_df[['group_members','upcoming_meetings','past_meetings','group_reviews']] = meetup_df[['group_members','upcoming_meetings','past_meetings','group_reviews']].apply(lambda x: pd.to_numeric(x, errors='ignore'))
print meetup_df['group_members'].dtype
#meetup_df[['group_members','upcoming_meetings','past_meetings','group_reviews']] = meetup_df[['group_members','upcoming_meetings','past_meetings','group_reviews']].astype(int, raise_on_error=False)
#Replacing None with NAN, which works better with numpy and they are missing values anyway
meetup_df =  meetup_df.replace('None', np.nan)

#Now we want to seperate the upcoming meeting dates to Day and Date 
meetup_df["upcoming_meeting_day"] = meetup_df["upcoming_meeting_date"].str[0:3]
meetup_df["upcoming_meeting_date"] = meetup_df["upcoming_meeting_date"].str.replace('^.{0,4}','')

#Adding the year to the date
meetup_df["upcoming_meeting_date"] = meetup_df["upcoming_meeting_date"] + ', 2017'

# converting the date to a date time object 
# meetup_df["upcoming_meeting_date"] = meetup_df["upcoming_meeting_date"].apply(lambda x: pd.to_datetime(x, format='%b %d, %Y', errors='ignore'))

meetup_df.dtypes

meetup_df.describe()

#What is the most popular group in all of Meetup? 
meetup_df['group_members'].argmax()
meetup_df.iloc[1423]

# What is the average group membership? 

meetup_df['group_members'].mean()

meetup_df.groupby(by="category")["group_members"].mean().astype(int)

cats  = meetup_df["category"].unique()
meetup_df.groupby(by="category")["group_members"].mean()

meetup_df.groupby(by="category")["group_members"].mean().plot.bar()
plt.ylabel('<group members>')

# What is the most popular group in each catergory? 
rows_max =  meetup_df.groupby(by="category")["group_members"].apply(lambda x: x.argmax())
list_of_max_cat = []
for row_max in rows_max: 
     list_of_max_cat.append(meetup_df.iloc[row_max][['category', "group_name"]])
max = 0
for i in range(len(list_of_max_cat)):
    if list_of_max_cat[i][1] > max:
        max_name = list_of_max_cat[i][0]
    print list_of_max_cat[i][0] +  ":" , list_of_max_cat[i][1]
    

#How many groups are in each category
meetup_df['category'].value_counts()

meetup_df['category'].value_counts(ascending=True).plot.barh()
plt.ylabel('Number of Groups')

meetup_df.groupby('category')['group_members'].mean().plot.bar()
plt.ylabel('Averge Group Members')

# Is there a relationship between number of group members and past meetings, upcoming meetings, etc.
import seaborn as sns
comp_meetup_df = meetup_df.dropna(axis=0)
meetGrid = sns.PairGrid(comp_meetup_df)
#meetGrid.map(plt.scatter)
meetGrid.map_diag(plt.hist)
meetGrid.map_offdiag(plt.scatter)

meetup_df.corr()

# Many groups have no information about them. How are many are there?  
missing_meetup = meetup_df.isnull()

np.sum(missing_meetup)

len(meetup_df[(np.sum(missing_meetup, axis=1) == 9)])# completely missing data 

missing_groups = meetup_df[(np.sum(missing_meetup, axis=1) == 9)].groupby("category").size()
missing_groups

meetup_df[np.sum(missing_meetup, axis=1) == 8]["group_name"]

missing_groups = meetup_df[(np.sum(missing_meetup, axis=1) == 8)].groupby("category").size()
missing_groups

# What day of the week is most common 
meetup_df["upcoming_meeting_day"].value_counts()

# What about per category? 
#pd.set_option('display.max_rows', 200)
result = meetup_df.groupby("category")["upcoming_meeting_day"].value_counts()
meetup_df.groupby(['category']).apply(lambda x: x['upcoming_meeting_day'].value_counts().index[0])

# What about the times of the day? 
meetup_df["upcoming_meeting_time"].value_counts()
meetup_df.groupby(['category']).apply(lambda x: x['upcoming_meeting_time'].value_counts().index[0])



