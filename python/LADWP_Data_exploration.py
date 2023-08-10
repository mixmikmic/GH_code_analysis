import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Wrangled_data_cap2.csv',encoding='latin-1',
                   index_col='Index',parse_dates=True).iloc[:,1:]
data.head()

# group features by day dataframe with Maximum value days displayed by feature
day1 = data.iloc[:,:2].groupby(pd.Grouper(freq='1D')).aggregate(np.sum)
weather = data[data['Temp']!=0].iloc[:,2:5].groupby(pd.Grouper(freq='1D')).aggregate(np.mean)
weekdays = data.iloc[:,5:6].groupby(pd.Grouper(freq='1D')).aggregate(np.max)
day1 = day1.merge(weather,right_index=True,left_index=True).merge(weekdays,
                 right_index=True,left_index=True).fillna(0)

for i in day1.iloc[:,:-2]:
    print('---------------\n','Max',i)
    print(day1[day1[i]== day1[i].max()])
    
print('\n-----------------Descriptive Statistics')
print(day1.describe().iloc[:,:-2])

#restructure weekdays df to allow easy plotting
weekdays = pd.DataFrame(day1['Demand(MW)'].values,index=day1.DofWk,columns=['Demand(MW)'])
weekdays = weekdays[weekdays['Demand(MW)']!=0]
weekdays.index=weekdays.index.map(str)

#grouped by week dataframe with Maximum value days displayed by feature
day7 = data.iloc[:,:2].groupby(pd.Grouper(freq='7D')).aggregate(np.sum)
weather7 = data[data['Temp']!=0].iloc[:,2:4].groupby(pd.Grouper(freq='7D')).aggregate(np.mean)
holidays7 = data.iloc[:,4:5].groupby(pd.Grouper(freq='7D')).aggregate(np.max)
day7 = day7.merge(weather7,right_index=True,left_index=True).merge(holidays7,
                 right_index=True,left_index=True)

for i in day7.iloc[:,:-1]:
    print('---------------\n','Max',i)
    print(day7[day7[i]== day7[i].max()])
    
print('\n-----------------Descriptive Statistics')
print(day1.describe().iloc[:,:-2])

#Demand Over Time Visualization
titles= ['Weekly Counts','Daily Counts','Day of the Week Counts (7= Sun)',
         'Hourly Counts']

for i,t in zip([day7['Demand(MW)'],day1['Demand(MW)'],weekdays,data['Demand(MW)']],titles):
    try:
        i = i.to_frame('Demand(MW)')
        plt.scatter(i.index,i['Demand(MW)'])
        plt.xticks(rotation=45)
        plt.title(t)
        plt.ylabel('Demand(MW)')
        plt.xlabel('Time Interval')
        plt.show()
        print(i.describe())
    except:
        plt.scatter(i.index,i['Demand(MW)'])
        plt.xticks(rotation=45)
        plt.title(t)
        plt.ylabel('Demand(MW)')
        plt.xlabel('Time Interval')
        plt.show()
        print(i.describe())

#Mean Demand by Day of the Week
daycomp = data.groupby(by='DofWk').aggregate(np.mean)
plt.bar(daycomp.index,daycomp['Demand(MW)'],tick_label=['Mon', 'Tues','Wed','Thurs','Fri','Sat','Sun'])
plt.ylabel('Average Demand(MW)')
plt.xlabel('Day of the Week')
plt.title('Day of the Week Averages (7= Sun)')
plt.show()
print(daycomp.iloc[:,:-2])

#Mean Demand by Temperature
meantemp = data.iloc[:,:3].groupby(by='Temp').aggregate(np.mean)

plt.scatter(meantemp.index,meantemp['Demand(MW)'])
plt.xlabel('Temperature (F)')
plt.ylabel('Demand(MW)')
plt.ylim((0, meantemp['Demand(MW)'].max()+100))
plt.title('Average Demand by Temperature')
plt.show()
print('Pearsons R =',np.corrcoef(meantemp.index,meantemp['Demand(MW)'])[1,0])

#holiday comparison
holidaycomp = data.groupby(by='holiday').aggregate(np.mean)
plt.bar(holidaycomp.index,holidaycomp['Demand(MW)'],tick_label=['No','Yes'])
plt.ylabel('Average Demand(MW)')
plt.xlabel('Hoilday?')
plt.title('Average Demand by Holiday')
plt.show()
print(holidaycomp.iloc[:,:-2])

#Demand by Daylight
lightcomp = data.groupby(by='daylight').aggregate(np.mean)
plt.bar(lightcomp.index,lightcomp['Demand(MW)'],tick_label=['Night Time','Day Time'])
plt.ylabel('Average Demand(MW)')
plt.xlabel('Daylight?')
plt.title('Average Demand by Daylight')
plt.show()
print(lightcomp.iloc[:,:-2])

#Impute Missing values for Demand and Temperature Data using Linear Interpolation
data.info()
print('-------------------------Initial Dataframe info')
for i in ['Demand(MW)','Temp']:
    data[i] = data[i].replace(to_replace = 0,value=np.NaN)
    missing = data[data[i].isnull()==True]
    data[i] = data[i].interpolate()
    filled = data.loc[missing.index]
data.info()
print('-------------------------Missing Data Imputed Dataframe info')

#---------------create lag features
lags = []
for j in range(30):
    time = []
    daylag = []
    interval = j+1
    for i in data.index[24*interval:]:
        dayago = i-np.timedelta64(1*interval,'D')
        daydemand = data.loc[dayago,'Demand(MW)']
        time.append(i)
        daylag.append(daydemand)
    lagdf = pd.DataFrame(daylag,index=time,columns=['dayLag%s'%interval])
    lags.append(lagdf)

#Merge lag features into original Data
lagdata = data
for i in lags:
    lagdata = lagdata.merge(i,right_index=True,left_index=True,how='outer')
lagdata.info()

#Create Correlation matrix for all features
corr = lagdata.corr()
f, ax = plt.subplots(figsize=(30, 12))
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.01)
f.subplots_adjust(top=0.93)
t= f.suptitle('Correlation Heatmap', fontsize=18)
plt.show()

