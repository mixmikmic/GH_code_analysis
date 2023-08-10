import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
get_ipython().magic('matplotlib inline')
import seaborn as sns; sns.set()

# Load the everyday record count data from spark analysis. 
gap_count = pd.read_csv('count.csv')

# Show the information that can get from the dataset.
gap_count.head()

# As the date type is string, the first step is to change it into timestamp.
time = gap_count['Date']
time = map(lambda x:x[:-2] + '20' + x[-2:],time)#x = x[:-2] + '20' + x[-2:]
# date = map(lambda x:datetime.strptime(x, '%m/%d/%Y',time))
gap_count['Date'] = time
print gap_count.head()

gap_count['Date_new'] = map(lambda x: datetime.strptime(x, '%m/%d/%Y'), gap_count['Date'])
print gap_count.head()
# Now get a new column called "Date_new".

# Count the total number of record for every day.
count_date = gap_count.groupby('Date_new').sum()

# Plot the whole year data. Show the trend and regularity.
ctdate = count_date.index
ctbus  = count_date.Count

#ctdate0 = map(lambda x: datetime.strptime(x, "-%d-%m-%Y").date(), ctdate)

fig = plt.figure(figsize=(20,15))
ax = plt.subplot(111)
ax.bar(ctdate, ctbus, width=1)
ax.xaxis_date()
ax.set_xlabel('Date', fontsize = 15)
ax.set_ylabel('Record Count', fontsize = 15)
ax.set_title('Bus Record Count For Everyday In A Year', fontsize =15)

plt.savefig('count_date.png')

# Here plot the data for every month.
ctmonth = map(lambda x: x.month, ctdate)
count_date2 = count_date.copy()
count_date2['month'] = ctmonth
count_date2 = count_date2.iloc[1:]
monthlist = ['January','February','March','April','May','June','July','August','September','October','November','December']

fig = plt.figure(figsize=(18,18))
for i in range(1,13):
    count_tmp = count_date2[count_date2['month'] == i]
    #fig = plt.figure(figsize=(12,8))
    ctdatetmp = count_tmp.index
    ctbustmp  = count_tmp.Count
    
    
    ax = fig.add_subplot(6,4,i)
    ax.bar(ctdatetmp, ctbustmp, width=1)
    ax.xaxis_date()
    plt.setp(plt.gca().get_xticklabels(),rotation=45) 
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Record Count')
    ax.set_title('Bus Record Count For Everyday In '+monthlist[i-1])

fig.tight_layout()

ctmonth = map(lambda x: x.month, ctdate)
count_date3 = count_date.copy()
count_date3['month'] = ctmonth
count_date3 = count_date3.iloc[1:]
monthlist = ['January','February','March','April','May','June','July','August','September','October','November','December']

fig = plt.figure(figsize=(18,18))
for i in range(1,13):
    count_tmp = count_date3[count_date3['month'] == i]
    #fig = plt.figure(figsize=(12,8))
    ctdatetmp = count_tmp.index
    ctbustmp  = count_tmp.Count
    
    
    ax = fig.add_subplot(6,4,i)
    ax.plot(ctdatetmp, ctbustmp)
    ax.xaxis_date()
    plt.setp(plt.gca().get_xticklabels(),rotation=45) 
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Record Count')
    ax.set_title('Bus Record Count For Everyday In '+ monthlist[i-1])

fig.tight_layout()

# Here we get the average of every month record.
ctmonth = map(lambda x: x.month, ctdate)
count_date4 = count_date.copy()
count_date4['month'] = ctmonth
count_date4 = count_date4.iloc[1:]
monthlist = ['January','February','March','April','May','June','July','August','September','October','November','December']

for i in range(1,13):
    count_tmp = count_date3[count_date3['month'] == i]
    ctdatetmp = count_tmp.index
    ctbustmp  = count_tmp.Count
    ctbustmp_mean = np.mean(ctbustmp)
    #print "The average of record count for " + monthlist[i-1] + " is " + str(round(ctbustmp_mean,2))
    print 'The average of record count for {} is {}'.format(monthlist[i-1], str(round(ctbustmp_mean,2)))

