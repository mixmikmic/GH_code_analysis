get_ipython().magic('matplotlib inline')
import pandas as pd
import seaborn as sns

get_ipython().system('wget -P data/ https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2016/06/NHS-111-from-Aug10-to-Nov16-web-file.csv')

#Detailed Excel data files (require parsers...)
#MDS-Web-File-National-November-2016.xlsx
#MDS-Web-File-North-November-2016.xlsx
#MDS-Web-File-London-November-2016.xlsx
#MDS-Web-File-Midlands-East-England-November-2016.xlsx
#!wget -P data/ https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2016/06/MDS-Web-File-South-November-2016.xlsx

df=pd.read_csv('data/NHS-111-from-Aug10-to-Nov16-web-file.csv',skiprows=4)
df.head()

df.columns

#Generate a pandas period for time series indexing purposes
#Create a date from each month (add the first date of the month) then set to period
#Note: if the datetime is an index, drop the .dt.
df['_period']=pd.to_datetime(('01-'+df['Periodname']),format='%d-%b-%y').dt.to_period('M')
df[['Yearnumber','Periodname','_period']].head()

#The groupby returns an index with period and provider;
#Find total number of recommendations to A&E in each group
#unstack on provider, fill NA with 0
df.groupby(['_period','Provider Name'])['Nhs1 Recommend To Ae SUM'].sum().unstack("Provider Name").fillna(0).plot();

#Plot the total A&E recommendations
df.groupby(['_period'])['Nhs1 Recommend To Ae SUM'].sum().fillna(0).plot()

#Chart for all recommendation columns
cols=[c for c in df.columns if c.startswith('Nhs1 Rec')]
sns.set_palette("Set2", len(cols))
pd.melt(df,id_vars='_period',value_vars=cols
       ).groupby(['_period','variable']).sum().unstack(1).fillna(0).plot();

#A&E referrals: https://www.england.nhs.uk/statistics/statistical-work-areas/ae-waiting-times-and-activity/statistical-work-areasae-waiting-times-and-activityae-attendances-and-emergency-admissions-2016-17/
get_ipython().system('wget -P data/ https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2016/06/Quarterly-timeseries-Nov-2016.xls')

df2=pd.read_excel('data/Quarterly-timeseries-Nov-2016.xls',skiprows=16,na_values='-').dropna(how='all')
df2['Year'].fillna(method='ffill',inplace=True)
#First group of cols are A&E attendances. I really need to sort out how to read multirow headers!
#Tidy data to remove empty rows at end of table, empty columns
df2.dropna(axis=0,subset=['Quarter'],inplace=True)
df2.dropna(axis=1,how='all',inplace=True)
df2.head()

df2[['Type 1 Departments - Major A&E',
     'Type 2 Departments - Single Specialty',
     'Type 3 Departments - Other A&E/Minor Injury Unit']].plot();

df2['period']=df2['Year']+' '+df2['Quarter']

ax=df2[['Type 1 Departments - Major A&E',
     'Type 2 Departments - Single Specialty',
     'Type 3 Departments - Other A&E/Minor Injury Unit']].plot(logy=True, xticks=df2.index, rot=90)
#ax.set_xticklabels(df2['period'])
ticks = ax.xaxis.get_ticklocs()
ax.xaxis.set_ticks(ticks[::8])
ax.xaxis.set_ticklabels(df2['period'][::8]);

def getMonthYear(row):
    month=row['Quarter'].split(':')[1].split('-')[0].strip()
    year=int(row['Year'].split('-')[0])
    if month in ['Jan']:
        year= year+1
    #Following the conversion, the _quarter year specifies the calendar year in which the financial year ends
    return pd.to_datetime("01-{}-{}".format(month[:3],year),format='%d-%b-%Y')

df2['_quarter']=pd.PeriodIndex(df2.apply(getMonthYear,axis=1), freq='Q-MAR')
df2[['Year','Quarter','_quarter']].head()
#Note the syntax - the _quarter year specifies the financial end year

#Review what the start and end dates of the corresponding periods are
tmp=df2.set_index('_quarter')
tmp['qstart']= tmp.index.asfreq('D', 's')
tmp['qend']=tmp.index.asfreq('D', 'e')
tmp[['Year','Quarter','qstart','qend']].head()

#The (calendar) year looks wrong to me in the below
#Need to somehow emphasis the year is the end calendar year of the corresponding (plotted) financial year?
df2.set_index('_quarter')[['Type 1 Departments - Major A&E',
     'Type 2 Departments - Single Specialty',
     'Type 3 Departments - Other A&E/Minor Injury Unit']].plot(logy=True, rot=90);



