from pandas import Series, DataFrame
import pandas as pd

ser = Series((range(51,100,2)))
#Series([x for x in range(51,100) if x%2==1])

list(ser[:5].values)

list(ser.values[:5])

ser[(ser>70) & (ser <80)]
ser[ser>70][ser<80]

[(x in ser.values) or (x in ser) for x in [0,96,89,6]]

sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = Series(sdata)
states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = Series(sdata, index=states)
obj5 = obj3 + obj4
obj5

list(obj5[obj5.notnull()].index)

df = pd.read_csv('/Users/spandanbrahmbhatt/Downloads/Chicago_crime.csv',index_col=0).reset_index(drop=True)

df.head()

len(df.IUCR.unique())

df[df['Case Number'] == 'HL274719']['Location Description'].values[0]

100*len(df[df.Arrest==True])/len(df)

len(df[df.Arrest==True])

year = df.Year.unique()
arrest_year = {}

for i in range(len(year)):
    arrest_year.setdefault(year[i],0)
    df_new = df[df['Year']==year[i]]
    num = len(df_new[df_new['Arrest']=='True'])
    arrest_year[year[i]] = num

df[df.Arrest==True]['Year'].value_counts()/df['Year'].value_counts()*100



df.ix[0,'Arrest']==False





df['Ward'].describe()



# Reading a CSV

import pandas as pd
df = pd.read_csv('/Users/spandanbrahmbhatt/Documents/DataAnalysisPython/data/stock_data/data/stocks/AAIT.csv')

df.head()

from datetime import datetime,date

df['day'] = df['Date'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d').date().strftime('%A'))

df['Date'] = df['Date'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d').date())

from glob import glob

all_df = []
for fi in glob('/Users/spandanbrahmbhatt/Documents/DataAnalysisPython/data/stock_data/data/stocks/*.csv'):
    name = fi.split('/')[-1].split('.')[0]
    df = pd.read_csv(fi)
    df['name'] = name
    all_df.append(df)
f_df = pd.concat(all_df,ignore_index=True)

f_df.head()

f_df['name'].value_counts().sort_index()







import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from datetime import datetime,date,timedelta
get_ipython().magic('matplotlib inline')

df = pd.read_csv('/Users/spandanbrahmbhatt/Documents/DataAnalysisPython/data/stock_data/data/stocks/AAIT.csv')

df.head(2)

new_df = df[['Date','High']]

new_df.loc[:,'Date'] = new_df['Date'].apply(lambda x : datetime.strptime(x,'%Y-%m-%d').strftime('%Y'))



plt.subplots(figsize=(20,8))
sns.set_style('whitegrid')
ax = sns.boxplot(x='Date',y='High',data=new_df,color='#3778bf')
ax.set_title('Distribution of stocks of AAIT over years')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
ax.xaxis.get_label().set_fontsize(30)
ax.yaxis.get_label().set_fontsize(30)
ax.title.set_fontsize(36)
ax.tick_params(axis='x', which='major',labelsize=24)
ax.tick_params(axis='y', which='major',labelsize=24)
#plt.ylim((0.8*new_df['High'].min(),1.2*new_df['High'].max()))

us_election = pd.read_csv('/Users/spandanbrahmbhatt/Downloads/usa-2016-presidential-election-by-county.csv',error_bad_lines=False,sep=';')

us_election.head()

candidates_df = us_election[['ST','Clinton H','Johnson G','Stein J','Trump D']]

candidates_df.head()

small_candidates_df= candidates_df[candidates_df.ST.isin(candidates_df.ST.unique()[-10:])]

small_candidates_df.head()

candidates_votes = small_candidates_df.groupby('ST').sum()

candidates_votes = candidates_votes.reset_index()

candidates_votes.head()

candidates_votes.head()

all_candidate = []
for can_name in 'Clinton H', 'Johnson G', 'Stein J', 'Trump D':
    single_candidate = candidates_votes[['ST',can_name]].rename(columns={can_name:'votes'})
    single_candidate['name'] = can_name
    all_candidate.append(single_candidate)
final_candidate_df = pd.concat(all_candidate)

final_candidate_df.head(2)

plt.subplots(figsize=(20,8))
sns.barplot(x='ST',y='votes',hue='name',data=final_candidate_df)



