#Download data file
get_ipython().system('wget -P /Users/ajh59/Documents/parlidata/ https://zenodo.org/record/579712/files/senti_post_v2.csv')

#Install some dependencies
get_ipython().system('pip3 install pydrill')
get_ipython().system('pip3 install pandas')
get_ipython().system('pip3 install matplotlib')

#Import necessary packages
import pandas as pd
from pydrill.client import PyDrill

#Set the notebooks up for inline plotting
get_ipython().magic('matplotlib inline')

#Get a connection to the Apache Drill server
drill = PyDrill(host='localhost', port=8047)

#Test the setup
drill.query(''' SELECT * from dfs.tmp.`/senti_post_v2.parquet` LIMIT 3''').to_dataframe()

#Get Parliament session dates from Parliament API
psd=pd.read_csv('http://lda.data.parliament.uk/sessions.csv?_view=Sessions&_pageSize=50')
psd

def getParliamentDate(session):
    start=psd[psd['display name']==session]['start date'].iloc[0]
    end=psd[psd['display name']==session]['end date'].iloc[0]
    return start, end

getParliamentDate('2015-2016')

#Check the columns in the Hansard dataset, along with example values
df=drill.query(''' SELECT * from dfs.tmp.`/senti_post_v2.parquet` LIMIT 1''').to_dataframe()
print(df.columns.tolist())
df.iloc[0]

# Example of count of speeches by person in the dataset as a whole
q='''
SELECT proper_name, COUNT(*) AS number 
FROM dfs.tmp.`/senti_post_v2.parquet`
GROUP BY proper_name
'''

df=drill.query(q).to_dataframe()
df.head()

# Example of count of speeches by gender in the dataset as a whole
q="SELECT gender, count(*) AS `Number of Speeches` FROM dfs.tmp.`/senti_post_v2.parquet` GROUP BY gender"
drill.query(q).to_dataframe()

#Query within session
session='2015-2016'

start,end=getParliamentDate(session)
q='''
SELECT '{session}' AS session, gender, count(*) AS `Number of Speeches`
FROM dfs.tmp.`/senti_post_v2.parquet`
WHERE speech_date>='{start}' AND speech_date<='{end}'
GROUP BY gender
'''.format(session=session, start=start, end=end)

drill.query(q).to_dataframe()

#Count number of speeches per person
start,end=getParliamentDate(session)
q='''
SELECT '{session}' AS session, gender, mnis_id, count(*) AS `Number of Speeches`
FROM dfs.tmp.`/senti_post_v2.parquet`
WHERE speech_date>='{start}' AND speech_date<='{end}'
GROUP BY mnis_id, gender
'''.format(session=session, start=start, end=end)

drill.query(q).to_dataframe().head()

# Example of finding the average number of speeches per person by gender in a particular session
q='''
SELECT AVG(gcount) AS average, gender, session
FROM (SELECT '{session}' AS session, gender, mnis_id, count(*) AS gcount
        FROM dfs.tmp.`/senti_post_v2.parquet`
        WHERE speech_date>='{start}' AND speech_date<='{end}'
        GROUP BY mnis_id, gender)
GROUP BY gender, session
'''.format(session=session, start=start, end=end)

drill.query(q).to_dataframe()

#Note - the average is returned as a string not a numeric

#We can package that query up in a Python function
def avBySession(session):
    start,end=getParliamentDate(session)
    q='''SELECT AVG(gcount) AS average, gender, session FROM (SELECT '{session}' AS session, gender, mnis_id, count(*) AS gcount
FROM dfs.tmp.`/senti_post_v2.parquet`
WHERE speech_date>='{start}' AND speech_date<='{end}'
GROUP BY mnis_id, gender) GROUP BY gender, session
'''.format(session=session, start=start, end=end)
    dq=drill.query(q).to_dataframe()
    #Make the average a numeric type...
    dq['average']=dq['average'].astype(float)
    return dq

avBySession(session)

#Loop through sessions and create a dataframe containing gender based averages for each one
overall=pd.DataFrame()
for session in psd['display name']:
    overall=pd.concat([overall,avBySession(session)])

#Tidy up the index
overall=overall.reset_index(drop=True)

overall.head()

#Reshape the dataset
overall_wide = overall.pivot(index='session', columns='gender')
#Flatten the column names
overall_wide.columns = overall_wide.columns.get_level_values(1)
overall_wide

overall_wide.plot(kind='barh');

overall_wide.plot();

# Example of finding the average number of speeches per person by party in a particular session
# Simply tweak the query we used for gender...
q='''
SELECT AVG(gcount) AS average, party, session
FROM (SELECT '{session}' AS session, party, mnis_id, count(*) AS gcount
        FROM dfs.tmp.`/senti_post_v2.parquet`
        WHERE speech_date>='{start}' AND speech_date<='{end}'
        GROUP BY mnis_id, party)
GROUP BY party, session
'''.format(session=session, start=start, end=end)

drill.query(q).to_dataframe()

def avByType(session,typ):
    start,end=getParliamentDate(session)
    q='''SELECT AVG(gcount) AS average, {typ}, session
        FROM (SELECT '{session}' AS session, {typ}, mnis_id, count(*) AS gcount
            FROM dfs.tmp.`/senti_post_v2.parquet`
            WHERE speech_date>='{start}' AND speech_date<='{end}'
            GROUP BY mnis_id, {typ})
        GROUP BY {typ}, session
'''.format(session=session, start=start, end=end, typ=typ)
    dq=drill.query(q).to_dataframe()
    #Make the average a numeric type...
    dq['average']=dq['average'].astype(float)
    return dq

def avByParty(session):
    return avByType(session,'party')

avByParty(session)

# Create a function to loop through sessions and create a dataframe containing specified averages for each one
# Note that this just generalises and packages up the code we had previously
def pivotAndFlatten(overall,typ):
    #Tidy up the index
    overall=overall.reset_index(drop=True)
    overall_wide = overall.pivot(index='session', columns=typ)
    
    #Flatten the column names
    overall_wide.columns = overall_wide.columns.get_level_values(1)
    return overall_wide

def getOverall(typ):
    overall=pd.DataFrame()
    for session in psd['display name']:
        overall=pd.concat([overall,avByType(session,typ)])

    return pivotAndFlatten(overall,typ)

overallParty=getOverall('party')

overallParty.head()

#Note that the function means it's now just as easy to query on another single column
getOverall('party_group')

overallParty.plot(kind='barh', figsize=(20,20));

parties=['Conservative','Labour']

overallParty[parties].plot();

def avByGenderAndParty(session):
    start,end=getParliamentDate(session)
    q='''SELECT AVG(gcount) AS average, gender, party, session
        FROM (SELECT '{session}' AS session, gender, party, mnis_id, count(*) AS gcount
            FROM dfs.tmp.`/senti_post_v2.parquet`
            WHERE speech_date>='{start}' AND speech_date<='{end}'
            GROUP BY mnis_id, gender, party)
        GROUP BY gender, party, session
'''.format(session=session, start=start, end=end)
    dq=drill.query(q).to_dataframe()
    #Make the average a numeric type...
    dq['average']=dq['average'].astype(float)
    return dq


gp=avByGenderAndParty(session)
gp

gp_overall=pd.DataFrame()

for session in psd['display name']:
    gp_overall=pd.concat([gp_overall,avByGenderAndParty(session)])

#Pivot table it more robust than pivot - missing entries handled with NA
#Also limit what parties we are interested in
gp_wide = gp_overall[gp_overall['party'].isin(parties)].pivot_table(index='session', columns=['party','gender'])

#Flatten column names
gp_wide.columns = gp_wide.columns.droplevel(0)
    
gp_wide

gp_wide.plot(figsize=(20,10));

gp_wide.plot(kind='barh', figsize=(20,10));

# Go back to the full dataset, not filtered by party
gp_wide = gp_overall.pivot_table(index='session', columns=['party','gender'])

#Flatten column names
gp_wide.columns = gp_wide.columns.droplevel(0)

gp_wide.head()

sp_wide = gp_wide.reset_index().melt(id_vars=['session']).pivot_table(index=['session','party'], columns=['gender'])

#Flatten column names
sp_wide.columns = sp_wide.columns.droplevel(0)

sp_wide#.dropna(how='all')

#Sessions when F spoke more, on average, then M
#Recall, this data has been previously filtered to limit data to Con and Lab

#Tweak the precision of the display
pd.set_option('precision',3)

sp_wide[sp_wide['Female'].fillna(0) > sp_wide['Male'].fillna(0) ]



