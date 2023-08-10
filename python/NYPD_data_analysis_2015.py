import pandas as pd
import matplotlib.pyplot as plt
import dateutil.parser
import datetime
get_ipython().magic('matplotlib inline')

# Source: http://www.nyc.gov/html/nypd/html/analysis_and_planning/stop_question_and_frisk_report.shtml
df = pd.read_csv('2015.csv', parse_dates=['datestop'])

df.head()

selection = df[['perobs', 'crimsusp', 'perstop', 'explnstp', 'arstmade', 'sumissue', 'sumoffen', 'frisked', 'rf_vcrim', 'rf_othsw', 'rf_attir', 'pf_hands', 'pf_wall', 'pf_grnd', 'pf_drwep', 'pf_ptwep', 'pf_baton', 'pf_hcuff', 'pf_other', 'pf_pepsp', 'detailCM', 'sex', 'race', 'age', 'state', 'city']]
selection.head()

selection['city'].value_counts()

selection['city'].value_counts().plot(kind='barh')

# dateutil.parser.parse("1012015")
# http://stackoverflow.com/questions/502726/converting-date-between-dd-mm-yyyy-and-yyyy-mm-dd
dateutil.parser.parse(datetime.datetime.strptime("1012015", "%m%d%Y").strftime("%Y-%m-%d"))

def time_conversion(date_str):
    return dateutil.parser.parse(datetime.datetime.strptime(date_str, "%m%d%Y").strftime("%Y-%m-%d"))

df.index = df['datestop'].apply(time_conversion)

df.head()

df['year'].count()

# fig, ax = plt.subplots(figsize=(10,5))
# ax = df.resample('m')['datestop'].count().plot(ax = ax)

# plt.savefig("ugly_graph.pdf", transparent=True)

selection['race'].value_counts()

selection['race'].value_counts()

# B = Black, Q = Hispanic, W = White, P = Black-Hispanic, A = Asian, 
# Z = Other, U = X? = Unknown, I = American Indian/Alaska Native

# NYC Population by race:
# White: 33%
# Black: 26%
# Hispanic: 26%
# Asian: 13%
# Other: 2%
#Source: http://furmancenter.org/files/sotc/The_Changing_Racial_and_Ethnic_Makeup_of_New_York_City_Neighborhoods_11.pdf

selection.groupby(by='race')['explnstp'].value_counts()

selection.groupby(by='race')['frisked'].value_counts()

black_df = selection[selection['race']=="B"]
black_df['frisked'].value_counts().plot(kind='pie', radius=0.02)
plt.axis('equal')

white_df = selection[selection['race']=="W"]
white_df['frisked'].value_counts().plot(kind='pie')
plt.axis('equal')

Asian_df = selection[selection['race']=="A"]
Asian_df['frisked'].value_counts().plot(kind='pie')
plt.axis('equal')

hispanic_df = selection[selection['race']=="Q"]
hispanic_df['frisked'].value_counts().plot(kind='pie')
plt.axis('equal')

black_df['frisked'].value_counts().plot(kind='pie', radius=1)
hispanic_df['frisked'].value_counts().plot(kind='pie', radius=0.65,)
white_df['frisked'].value_counts().plot(kind='pie', radius =0.46)
plt.axis('equal')
plt.savefig('FriskedNEW.pdf', transparend=True)

# http://pandas.pydata.org/pandas-docs/stable/visualization.html

selection.groupby(by='race')['arstmade'].value_counts()

selection['arstmade'].value_counts()

selection.groupby(by='race')['sumissue'].value_counts()

selection['sumissue'].value_counts()

selection.groupby(by='race')['sumoffen'].value_counts()

black_df = selection[selection['race']=="B"]
black_df['sumoffen'].value_counts()

selection['detailCM'].value_counts()

def police_force_count(columnlist):
    for columnname in columnlist:
        print(selection[columnname].value_counts())

police_force_count(['pf_hands', 'pf_wall', 'pf_grnd', 'pf_drwep', 'pf_ptwep', 'pf_baton', 'pf_hcuff', 'pf_other', 'pf_pepsp'])

selection.groupby(by='race')['pf_hcuff'].value_counts()

selection.groupby(by='race')['pf_hands'].value_counts()

selection.groupby(by='race')['pf_wall'].value_counts()



