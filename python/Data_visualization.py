def read_311_data(datafile):
    import pandas as pd
    import numpy as np
    
    #Add the fix_zip function
    def fix_zip(input_zip):
        try:
            input_zip = int(float(input_zip))
        except:
            try:
                input_zip = int(input_zip.split('-')[0])
            except:
                return np.NaN
        if input_zip < 10000 or input_zip > 19999:
            return np.NaN
        return str(input_zip)
    
    #Read the file
    df = pd.read_csv(datafile,index_col='Unique Key')
    
    #fix the zip
    df['Incident Zip'] = df['Incident Zip'].apply(fix_zip)
    
    #drop all rows that have any nans in them (note the easier syntax!)
    
    df = df.dropna(how='any')
    
    #get rid of unspecified boroughs
    df = df[df['Borough'] != 'Unspecified']
    
    #Convert times to datetime and create a processing time column
    
    import datetime
    df['Created Date'] = df['Created Date'].apply(lambda x:datetime.
                                                  datetime.
                                                  strptime(x,'%m/%d/%Y %H:%M'))
    df['Closed Date'] = df['Closed Date'].apply(lambda x:datetime.
                                                datetime.
                                                strptime(x,'%m/%d/%Y %H:%M'))
    df['processing_time'] =  df['Closed Date'].subtract(df['Created Date'])
    
    #Finally, get rid of negative processing times and return the final data frame
    
    df = df[df['processing_time']>=datetime.timedelta(0,0,0)]
    
    return df
    

datafile = "nyc_311_data_subset-2.csv"
data = read_311_data(datafile)

get_ipython().system('pip install gmplot --upgrade')


import gmplot
#gmap = gmplot.GoogleMapPlotter(40.7128, 74.0059, 8)


gmap = gmplot.GoogleMapPlotter.from_geocode("New York",10)

#Then generate a heatmap using the latitudes and longitudes
gmap.heatmap(data['Latitude'], data['Longitude'])

gmap.draw('incidents3.html')



get_ipython().run_line_magic('matplotlib', 'inline')


borough_group = data.groupby('Borough')
borough_group.size().plot(kind='bar')
#kind can be 'hist', 'scatter'

agency_group = data.groupby('Agency')
agency_group.size().plot(kind='bar')


agency_borough = data.groupby(['Agency','Borough'])
agency_borough.size().plot(kind='bar')

agency_borough.size().unstack().plot(kind='bar')


agency_borough = data.groupby(['Agency','Borough'])
agency_borough.size().unstack().plot(kind='bar',title="Incidents in each Agency by Borough",figsize=(15,15))

import pandas as pd
writers = pd.DataFrame({'Author':['George Orwell','John Steinbeck',
                                  'Pearl Buck','Agatha Christie'],
                        'Country':['UK','USA','USA','UK'],
                        'Gender':['M','M','F','F'],
                        'Age':[46,66,80,85]})
                        

writers

grouped = writers.groupby('Country')
grouped.first()
#grouped.last()
#grouped.sum()
#grouped.mean()
#grouped.apply(sum)

grouped.groups

grouped = writers.groupby(['Country','Gender'])
grouped.groups

def age_groups(df,index,col):
    print(index,col)
    if df[col].iloc[index] < 30:
        return 'Young'
    if df[col].iloc[index] < 60:
        return 'Middle'
    else:
        return 'Old'

writers['Age'].iloc[0]

grouped = writers.groupby(lambda x: age_groups(writers,x,'Age'))
grouped.groups

import numpy as np 

people = pd.DataFrame(np.random.randn(5, 5), columns=['a', 'b', 'c', 'd', 'e'], index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])
people

def GroupColFunc(df, ind, col):
    if df[col].loc[ind] > 0:
        return 'Group1'
    else:
        return 'Group2'

people.groupby(lambda x: GroupColFunc(people, x, 'a')).groups

print(people.groupby(lambda x: GroupColFunc(people, x, 'a')).mean())
print(people.groupby(lambda x: GroupColFunc(people, x, 'a')).std())

import datetime
data['yyyymm'] = data['Created Date'].apply(lambda x:datetime.datetime.strftime(x,'%Y%m'))

data['yyyymm']

date_agency = data.groupby(['yyyymm','Agency'])
date_agency.size().unstack().plot(kind='bar',figsize=(15,15))

data.groupby('Agency').size().sort_values(ascending=False)

data.groupby('Agency').size().sort_values(ascending=False).plot(kind='bar', figsize=(20,4))

agency_borough = data.groupby(['Agency', 'Borough']).size().unstack()

agency_borough

#We'll arrange the subplots in two rows and three columns. 
#Since we have only 5 boroughs, one plot will be blank
COL_NUM = 2
ROW_NUM = 3
import matplotlib.pyplot as plt
fig, axes = plt.subplots(ROW_NUM, COL_NUM, figsize=(12,12))

for i, (label, col) in enumerate(agency_borough.iteritems()): 
    ax = axes[int(i/COL_NUM), i%COL_NUM]
    col = col.sort_values(ascending=False)[:5] 
    col.plot(kind='barh', ax=ax)
    ax.set_title(label)

plt.tight_layout() 

for i, (label, col) in enumerate(agency_borough.iteritems()): 
    print(i,label,col)

grouped = data[['processing_time','Borough']].groupby('Borough')

grouped.describe()

import numpy as np
#The time it takes to process. Cleaned up
data['float_time'] =data['processing_time'].apply(lambda x:x/np.timedelta64(1, 'D'))

data

grouped = data[['float_time','Agency']].groupby('Agency')
grouped.mean().sort_values('float_time',ascending=False)

data['float_time'].hist(bins=50)





