get_ipython().magic('pylab inline')
import pandas as pd
import re

#read in the raw data
rawdata = pd.read_csv('Ebola_in_SL_Raw_WHO_Data.csv')
rawdata.iloc[1]

#parse the dates column
import dateutil

def parsedate(week_string):
    end_date_str = re.split(' to ', week_string)[1]
    return(dateutil.parser.parse(end_date_str))

rawdata['End Date'] = rawdata['EPI_WEEK (DISPLAY)'].apply(parsedate)
rawdata.head()

data = rawdata[rawdata['EBOLA_DATA_SOURCE (CODE)']=='PATIENTDB']
data = data[['End Date','Numeric']]
data.sort(columns='End Date', inplace=True)
data.dropna(inplace=True)
data['Timedelta'] = data['End Date']-data['End Date'].iloc[0]
data['Weeks'] = data['Timedelta'].apply(lambda a: pd.tslib.Timedelta(a).days/7)
data.set_index('Weeks', inplace=True)
data = data[['Numeric']]
data.columns=['New Reported Cases']
data['Cumulative Cases'] = data['New Reported Cases'].cumsum()

data.plot()

data.to_csv('Ebola_in_SL_Data.csv')



